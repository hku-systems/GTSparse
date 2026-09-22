from __future__ import annotations

from typing import Tuple

import torch

from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

from .metadata import GeometricTemplateMetadata, GeometricTemplateReverseEdge
from .runtime import (
    GeometricTemplateRuntime,
    build_full_runtime_from_coords as _build_full_runtime_from_coords,
    build_reverse_runtime_from_full_runtime as _build_reverse_runtime_from_full_runtime,
    build_runtime_from_coords as _build_runtime_from_coords,
    fp16_conv,
    fp32_conv,
)
from .tensor import tensor_metadata


BM = 128


def _conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: GeometricTemplateRuntime,
) -> torch.Tensor:
    if features.dtype == torch.float16:
        return fp16_conv(features, logical_weight, runtime)
    if features.dtype == torch.float32:
        return fp32_conv(features, logical_weight, runtime)
    raise TypeError(f"unsupported feature dtype {features.dtype}")


def build_subm_runtime_from_coords(
    x: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int = BM,
    sorted: bool = False,
) -> GeometricTemplateRuntime:
    return _build_runtime_from_coords(
        x,
        kernel_size,
        padding,
        dilation,
        max_bm=int(max_bm),
        sorted=bool(sorted),
    )


def build_full_runtime_from_coords(
    x: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int = BM,
    build_reverse_cache: bool = True,
    sorted: bool = False,
) -> GeometricTemplateRuntime:
    return _build_full_runtime_from_coords(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        max_bm=int(max_bm),
        build_reverse_cache=bool(build_reverse_cache),
        sorted=bool(sorted),
    )


def build_reverse_runtime_from_full_runtime(
    forward_runtime: GeometricTemplateRuntime,
    *,
    out_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int] | list[int],
    max_bm: int = BM,
    sorted: bool | None = None,
) -> GeometricTemplateRuntime:
    return _build_reverse_runtime_from_full_runtime(
        forward_runtime,
        out_coords,
        out_spatial,
        max_bm=int(max_bm),
        sorted=sorted,
    )


def subm_conv_tensor(
    x: GTSparseSparseConvTensor,
    logical_weight: torch.Tensor,
    *,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    max_bm: int = BM,
    sorted: bool = False,
) -> GTSparseSparseConvTensor:
    metadata = tensor_metadata(x)
    runtime = metadata.subm_runtime
    if runtime is None or bool(runtime.sorted) != bool(sorted):
        runtime = build_subm_runtime_from_coords(
            x,
            kernel_size,
            padding,
            dilation,
            max_bm=int(max_bm),
            sorted=bool(sorted),
        )
    out_features = _conv(x.features, logical_weight, runtime)
    out_metadata = GeometricTemplateMetadata(
        subm_runtime=runtime,
        reverse_chain=metadata.reverse_chain,
    )
    return x.replace_feature(out_features, metadata=out_metadata)


def full_conv_tensor(
    x: GTSparseSparseConvTensor,
    logical_weight: torch.Tensor,
    *,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    max_bm: int = BM,
    build_reverse: bool = True,
    sorted: bool = False,
) -> GTSparseSparseConvTensor:
    metadata = tensor_metadata(x)
    runtime = build_full_runtime_from_coords(
        x,
        kernel_size,
        stride,
        padding,
        dilation,
        max_bm=int(max_bm),
        build_reverse_cache=bool(build_reverse),
        sorted=bool(sorted),
    )
    out_features = _conv(x.features, logical_weight, runtime)
    if build_reverse:
        reverse_runtime = build_reverse_runtime_from_full_runtime(
            runtime,
            out_coords=x.indices,
            out_spatial=tuple(int(v) for v in x.spatial_shape),
            max_bm=int(max_bm),
            sorted=bool(sorted),
        )
        reverse_chain = (
            GeometricTemplateReverseEdge(runtime=reverse_runtime, coord_hashmap=x.coord_hashmap),
            *metadata.reverse_chain,
        )
    else:
        reverse_chain = ()
    out_metadata = GeometricTemplateMetadata(subm_runtime=None, reverse_chain=reverse_chain)
    return x.replace_sparse(
        new_features=out_features,
        new_coords=runtime.out_coords,
        new_spatial_shape=runtime.out_spatial,
        new_batch_size=x.batch_size,
        coord_hashmap=runtime.coord_hashmap,
        metadata=out_metadata,
    )


def inverse_conv_tensor(
    x: GTSparseSparseConvTensor,
    logical_weight: torch.Tensor,
    *,
    max_bm: int = BM,
) -> GTSparseSparseConvTensor:
    del max_bm
    metadata = tensor_metadata(x)
    if not metadata.reverse_chain:
        raise RuntimeError("inverse conv requires reverse metadata, but reverse_chain is empty")
    edge = metadata.reverse_chain[0]
    runtime = edge.runtime
    out_features = _conv(x.features, logical_weight, runtime)
    out_metadata = GeometricTemplateMetadata(
        subm_runtime=None,
        reverse_chain=metadata.reverse_chain[1:],
    )
    return x.replace_sparse(
        new_features=out_features,
        new_coords=runtime.out_coords,
        new_spatial_shape=runtime.out_spatial,
        new_batch_size=x.batch_size,
        coord_hashmap=edge.coord_hashmap,
        metadata=out_metadata,
    )
