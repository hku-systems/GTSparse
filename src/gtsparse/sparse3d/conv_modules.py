"""GTSparse 3D sparse convolution modules.

API-compatible with spconv v1. Two code paths:
  - Fallback: densify → F.conv3d → re-sparsify (works now, no CUDA kernels needed)
  - GTSparse:   expanded-worklist sparse convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union

from .sparse_tensor import GTSparseSparseConvTensor, IndiceData
from .reference_ops import reference_sparse_conv3d, reference_subm_conv3d
from .expanded_worklist import (
    build_expanded_worklist_subm,
    build_expanded_worklist_full,
    build_expanded_worklist_inverse,
    expanded_worklist_conv3d_cuda,
)


def _normalize_kernel(kernel_size) -> Tuple[int, ...]:
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size, kernel_size)
    return tuple(kernel_size)


def _normalize_padding(padding, kernel_size) -> Tuple[int, ...]:
    if isinstance(padding, int):
        return (padding, padding, padding)
    return tuple(padding)


def _normalize_stride(stride) -> Tuple[int, ...]:
    if isinstance(stride, int):
        return (stride, stride, stride)
    return tuple(stride)


def _normalize_dilation(dilation) -> Tuple[int, ...]:
    if isinstance(dilation, int):
        return (dilation, dilation, dilation)
    return tuple(dilation)


def _compute_output_shape(in_shape, kernel, stride, padding, dilation):
    return [
        (in_shape[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
        for i in range(3)
    ]


class GTSparseSubMConv3d(nn.Module):
    """Submanifold sparse 3D convolution. Output active set == input active set."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 indice_key=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_kernel(kernel_size)
        self.stride = _normalize_stride(stride)
        self.padding = _normalize_padding(padding, self.kernel_size)
        self.dilation = _normalize_dilation(dilation)
        self.groups = groups
        self.indice_key = indice_key

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        # Reference path for debugging: dense conv3d with inactive inputs kept at zero,
        # then gather only the active outputs.
        return reference_subm_conv3d(
            x,
            self.weight,
            self.bias,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class GTSparseSparseConv3d(nn.Module):
    """Regular (strided) sparse 3D convolution. Output active set may differ."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 indice_key=None, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_kernel(kernel_size)
        self.stride = _normalize_stride(stride)
        self.padding = _normalize_padding(padding, self.kernel_size)
        self.dilation = _normalize_dilation(dilation)
        self.groups = groups
        self.indice_key = indice_key

        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels // groups, *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        pairs, counts, out_coords, out_spatial = build_expanded_worklist_full(
            x, self.kernel_size, self.stride, self.padding, self.dilation)

        N_out = out_coords.size(0)
        weight_cl3d = self.weight.to(memory_format=torch.channels_last_3d)

        out_features = expanded_worklist_conv3d_cuda(
            x.features, weight_cl3d, self.bias,
            pairs, counts, N_out,
            active_ratio=x.features.size(0) / max(1, np.prod(x.spatial_shape) * x.batch_size),
        )

        out = GTSparseSparseConvTensor(out_features, out_coords, out_spatial, x.batch_size)
        out.indice_dict = dict(x.indice_dict)

        if self.indice_key is not None:
            out.indice_dict[self.indice_key] = IndiceData(
                in_indices=x.indices,
                out_indices=out_coords,
                in_spatial_shape=x.spatial_shape,
                out_spatial_shape=out_spatial,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )

        return out

def _same_active_set(x: GTSparseSparseConvTensor, y: GTSparseSparseConvTensor) -> bool:
    """Check if two sparse tensors share the same active set."""
    if x._index_grid is not None and y._index_grid is not None:
        if x._index_grid.data_ptr() == y._index_grid.data_ptr():
            return True
    return (x.spatial_shape == y.spatial_shape
            and x.batch_size == y.batch_size
            and x.features.size(0) == y.features.size(0)
            and torch.equal(x.indices, y.indices))


def sparse_add(x: GTSparseSparseConvTensor, y: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
    """Element-wise addition of two sparse tensors with the same active set."""
    assert _same_active_set(x, y), "SparseAdd requires identical active sets"
    out = GTSparseSparseConvTensor(
        x.features + y.features, x.indices, x.spatial_shape, x.batch_size,
        index_grid=x._index_grid)
    out.indice_dict = dict(x.indice_dict)
    return out


def sparse_cat(x: GTSparseSparseConvTensor, y: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
    """Concatenate features of two sparse tensors with the same active set."""
    assert _same_active_set(x, y), "SparseCat requires identical active sets"
    out = GTSparseSparseConvTensor(
        torch.cat([x.features, y.features], dim=1),
        x.indices, x.spatial_shape, x.batch_size,
        index_grid=x._index_grid)
    out.indice_dict = dict(x.indice_dict)
    return out


# PLACEHOLDER_INVERSE_CONV


class GTSparseSparseInverseConv3d(nn.Module):
    """Inverse (transposed) sparse 3D convolution, paired with a SparseConv3d via indice_key."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 indice_key=None, bias=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _normalize_kernel(kernel_size)
        self.indice_key = indice_key

        self.weight = nn.Parameter(torch.empty(
            in_channels, out_channels, *self.kernel_size
        ))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        assert self.indice_key is not None, "SparseInverseConv3d requires indice_key"
        pair = x.indice_dict.get(self.indice_key)
        assert pair is not None, f"indice_key '{self.indice_key}' not found in indice_dict"

        # Output active set = forward conv's input active set
        out_coords = pair.in_indices
        out_spatial = pair.in_spatial_shape
        N_out = out_coords.size(0)

        stride = pair.stride
        padding = pair.padding
        dilation = pair.dilation
        if stride is None or padding is None or dilation is None:
            stride = tuple(
                pair.in_spatial_shape[i] // max(pair.out_spatial_shape[i], 1)
                for i in range(3)
            )
            padding = tuple((k - 1) // 2 for k in self.kernel_size)
            dilation = (1, 1, 1)

        pairs, counts = build_expanded_worklist_inverse(
            x, out_coords, out_spatial,
            self.kernel_size, stride, padding, dilation)

        # Inverse conv weight: [C_in, C_out, kD, kH, kW]
        # GEMM kernel expects [C_out_gemm, C_in_gemm, kD, kH, kW] channels_last_3d
        # where C_in_gemm = inverse's input channels, C_out_gemm = inverse's output channels
        # So permute (1,0,2,3,4) to get [C_out, C_in, kD, kH, kW]
        w_transposed = self.weight.permute(1, 0, 2, 3, 4).contiguous()
        weight_cl3d = w_transposed.to(memory_format=torch.channels_last_3d)

        out_features = expanded_worklist_conv3d_cuda(
            x.features, weight_cl3d, self.bias,
            pairs, counts, N_out,
            active_ratio=x.features.size(0) / max(1, np.prod(x.spatial_shape) * x.batch_size),
        )

        out = GTSparseSparseConvTensor(out_features, out_coords, out_spatial, x.batch_size)
        out.indice_dict = dict(x.indice_dict)
        return out
