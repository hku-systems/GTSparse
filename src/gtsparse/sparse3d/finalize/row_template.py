from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from gtsparse import _C
from gtsparse.sparse3d.row_template_segmented import (
    SegmentedSubmCodebook,
    build_segmented_flat_tiled_runtime as _build_segmented_flat_tiled_runtime,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

_FP16_SETTING1_BASE_CONFIG_ID = 100
_FP32_SETTING1_BASE_CONFIG_ID = 200


@dataclass(frozen=True, slots=True)
class FinalizedRowTemplateDispatch:
    dtype_family: str
    setting: int
    config_id: Optional[int]
    k_ld_factor: Optional[int] = None
    n_ld_factor: Optional[int] = None
    k_ld_check: Optional[bool] = None
    n_ld_check: Optional[bool] = None


@dataclass(slots=True)
class FinalizedRowTemplateFlatTiledRuntime:
    out_rows: torch.Tensor
    input_rows_w1: torch.Tensor
    input_rows_w2: torch.Tensor
    input_rows_w9: torch.Tensor
    input_rows_w18: torch.Tensor
    input_rows_w27: torch.Tensor
    template_ids: torch.Tensor
    input_row_offsets: torch.Tensor
    padded_row_count: torch.Tensor
    n_out: int
    bm: int
    out_coords: Optional[torch.Tensor] = None
    out_spatial: Optional[Tuple[int, int, int]] = None
    coord_hashmap: Optional[torch.Tensor] = None

    @property
    def template_stride(self) -> int:
        return int(self.input_rows_w1.size(1))

    @property
    def num_tiles(self) -> int:
        return int(self.padded_row_count.item()) // int(self.bm)

    @property
    def input_segments(self) -> tuple[torch.Tensor, ...]:
        return (
            self.input_rows_w1,
            self.input_rows_w2,
            self.input_rows_w9,
            self.input_rows_w18,
            self.input_rows_w27,
        )


def _require_supported_feature_dtype(features: torch.Tensor) -> None:
    if features.dtype not in (torch.float16, torch.float32):
        raise TypeError("finalized row-template flat tiled kernel currently requires float16 or float32 features and weight")


def _require_flat_weight(features: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if weight.dim() != 2:
        raise ValueError("finalized row-template path requires pre-flattened 2D weight")
    if not weight.is_contiguous():
        raise ValueError("finalized row-template path requires contiguous flat weight")
    if weight.dtype != features.dtype:
        raise TypeError("features and weight must share dtype")
    expected_rows = 27 * int(features.size(1))
    if int(weight.size(0)) != expected_rows:
        raise ValueError(
            f"finalized row-template flat weight must have shape [{expected_rows}, Cout], "
            f"got {tuple(weight.shape)}"
        )
    return weight


def _finalized_row_template_flat_tiled_forward_unchecked(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: FinalizedRowTemplateFlatTiledRuntime,
    *,
    config_id: int,
) -> torch.Tensor:
    return _C.gtsparse3d_finalize_row_template_flat_tiled_forward(
        features,
        weight,
        runtime.out_rows,
        runtime.input_rows_w1,
        runtime.input_rows_w2,
        runtime.input_rows_w9,
        runtime.input_rows_w18,
        runtime.input_rows_w27,
        runtime.template_ids,
        runtime.input_row_offsets,
        runtime.padded_row_count,
        int(runtime.n_out),
        int(runtime.bm),
        int(config_id),
    )


def _select_fp16_setting1_fields(
    c_in: int,
    c_out: int,
) -> tuple[int, int, bool, int, bool]:
    if c_in % 16 == 0:
        k_case = 0
        k_ld_factor = 16
        k_ld_check = False
    elif c_in % 8 == 0:
        k_case = 1
        k_ld_factor = 16
        k_ld_check = True
    elif c_in % 4 == 0:
        k_case = 2
        k_ld_factor = 8
        k_ld_check = True
    elif c_in % 2 == 0:
        k_case = 3
        k_ld_factor = 4
        k_ld_check = True
    else:
        k_case = 4
        k_ld_factor = 2
        k_ld_check = True

    if c_out % 16 == 0:
        n_case = 0
        n_ld_factor = 16
        n_ld_check = False
    elif c_out % 8 == 0:
        n_case = 1
        n_ld_factor = 16
        n_ld_check = True
    elif c_out % 4 == 0:
        n_case = 2
        n_ld_factor = 8
        n_ld_check = True
    elif c_out % 2 == 0:
        n_case = 3
        n_ld_factor = 4
        n_ld_check = True
    else:
        n_case = 4
        n_ld_factor = 2
        n_ld_check = True

    return (
        _FP16_SETTING1_BASE_CONFIG_ID + k_case * 5 + n_case,
        k_ld_factor,
        k_ld_check,
        n_ld_factor,
        n_ld_check,
    )


def _select_fp16_setting1(c_in: int, c_out: int) -> FinalizedRowTemplateDispatch:
    config_id, k_ld_factor, k_ld_check, n_ld_factor, n_ld_check = _select_fp16_setting1_fields(c_in, c_out)
    return FinalizedRowTemplateDispatch(
        dtype_family="fp16",
        setting=1,
        config_id=config_id,
        k_ld_factor=k_ld_factor,
        n_ld_factor=n_ld_factor,
        k_ld_check=k_ld_check,
        n_ld_check=n_ld_check,
    )


def _select_fp32_setting1_fields(
    c_in: int,
    c_out: int,
) -> tuple[int, int, bool, int, bool]:
    if c_in % 16 == 0:
        k_case = 0
        k_ld_factor = 16
        k_ld_check = False
    elif c_in % 4 == 0:
        k_case = 1
        k_ld_factor = 16
        k_ld_check = True
    elif c_in % 2 == 0:
        k_case = 2
        k_ld_factor = 8
        k_ld_check = True
    else:
        k_case = 3
        k_ld_factor = 4
        k_ld_check = True

    if c_out % 16 == 0:
        n_case = 0
        n_ld_factor = 16
        n_ld_check = False
    elif c_out % 4 == 0:
        n_case = 1
        n_ld_factor = 16
        n_ld_check = True
    elif c_out % 2 == 0:
        n_case = 2
        n_ld_factor = 8
        n_ld_check = True
    else:
        n_case = 3
        n_ld_factor = 4
        n_ld_check = True

    return (
        _FP32_SETTING1_BASE_CONFIG_ID + k_case * 4 + n_case,
        k_ld_factor,
        k_ld_check,
        n_ld_factor,
        n_ld_check,
    )


def _select_fp32_setting1(c_in: int, c_out: int) -> FinalizedRowTemplateDispatch:
    config_id, k_ld_factor, k_ld_check, n_ld_factor, n_ld_check = _select_fp32_setting1_fields(c_in, c_out)
    return FinalizedRowTemplateDispatch(
        dtype_family="fp32",
        setting=1,
        config_id=config_id,
        k_ld_factor=k_ld_factor,
        n_ld_factor=n_ld_factor,
        k_ld_check=k_ld_check,
        n_ld_check=n_ld_check,
    )


def _select_finalize_row_template_config_id(
    *,
    c_in: int,
    c_out: int,
    dtype: torch.dtype,
) -> int:
    c_in = int(c_in)
    c_out = int(c_out)
    if dtype == torch.float16:
        if c_out % 64 == 0 and c_in % 32 == 0:
            return 1
        if c_in % 32 == 0 and c_out % 16 == 0:
            return 2
        return _select_fp16_setting1_fields(c_in, c_out)[0]
    if dtype == torch.float32:
        if c_out % 64 == 0 and c_in % 32 == 0:
            return 1
        if c_in % 32 == 0 and c_out % 16 == 0:
            return 2
        return _select_fp32_setting1_fields(c_in, c_out)[0]
    raise TypeError(f"finalized row-template dispatch does not support dtype={dtype}")


def select_finalize_row_template_dispatch(
    *,
    c_in: int,
    c_out: int,
    dtype: torch.dtype,
) -> FinalizedRowTemplateDispatch:
    c_in = int(c_in)
    c_out = int(c_out)
    if dtype == torch.float16:
        if c_out % 64 == 0 and c_in % 32 == 0:
            return FinalizedRowTemplateDispatch(dtype_family="fp16", setting=3, config_id=1)
        if c_in % 32 == 0 and c_out % 16 == 0:
            return FinalizedRowTemplateDispatch(dtype_family="fp16", setting=2, config_id=2)
        return _select_fp16_setting1(c_in, c_out)
    if dtype == torch.float32:
        if c_out % 64 == 0 and c_in % 32 == 0:
            return FinalizedRowTemplateDispatch(dtype_family="fp32", setting=3, config_id=1)
        if c_in % 32 == 0 and c_out % 16 == 0:
            return FinalizedRowTemplateDispatch(dtype_family="fp32", setting=2, config_id=2)
        return _select_fp32_setting1(c_in, c_out)
    raise TypeError(f"finalized row-template dispatch does not support dtype={dtype}")


def _convert_segmented_runtime(rt) -> FinalizedRowTemplateFlatTiledRuntime:
    return FinalizedRowTemplateFlatTiledRuntime(
        out_rows=rt.out_rows,
        input_rows_w1=rt.input_rows_w1,
        input_rows_w2=rt.input_rows_w2,
        input_rows_w9=rt.input_rows_w9,
        input_rows_w18=rt.input_rows_w18,
        input_rows_w27=rt.input_rows_w27,
        template_ids=rt.template_ids,
        input_row_offsets=rt.input_row_offsets,
        padded_row_count=rt.padded_row_count,
        n_out=int(rt.n_out),
        bm=int(rt.bm),
        out_coords=None,
        out_spatial=None,
        coord_hashmap=None,
    )


def build_finalize_row_template_flat_tiled_runtime(
    runtime: SegmentedSubmCodebook,
    *,
    bm: int,
) -> FinalizedRowTemplateFlatTiledRuntime:
    return _convert_segmented_runtime(_build_segmented_flat_tiled_runtime(runtime, bm=bm))


def build_finalize_row_template_flat_tiled_runtime_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
) -> FinalizedRowTemplateFlatTiledRuntime:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("finalized row-template flat tiled currently requires kernel_size=(3, 3, 3)")
    if tuple(int(v) for v in padding) != (1, 1, 1):
        raise ValueError("finalized row-template flat tiled currently requires padding=(1, 1, 1)")
    if tuple(int(v) for v in dilation) != (1, 1, 1):
        raise ValueError("finalized row-template flat tiled currently requires dilation=(1, 1, 1)")
    (
        out_rows,
        input_rows_w1,
        input_rows_w2,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        padded_row_count,
        coord_hashmap,
    ) = _C.gtsparse3d_finalize_row_template_flat_tiled_build_runtime_from_coords(
        st.indices.contiguous(),
        int(max_bm),
        torch.Tensor() if st.coord_hashmap is None else st.coord_hashmap,
    )
    return FinalizedRowTemplateFlatTiledRuntime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w2=input_rows_w2,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        padded_row_count=padded_row_count,
        n_out=int(st.indices.size(0)),
        bm=int(max_bm),
        out_coords=st.indices,
        out_spatial=tuple(int(v) for v in st.spatial_shape),
        coord_hashmap=coord_hashmap,
    )


def build_finalize_row_template_flat_tiled_full_runtime_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
    lookup_coord_hashmap: Optional[torch.Tensor] = None,
) -> FinalizedRowTemplateFlatTiledRuntime:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("finalized row-template flat tiled full builder currently requires kernel_size=(3, 3, 3)")
    stride = tuple(int(v) for v in stride)
    padding = tuple(int(v) for v in padding)
    dilation = tuple(int(v) for v in dilation)
    out_spatial = tuple(
        (int(st.spatial_shape[i]) + 2 * padding[i] - dilation[i] * (int(kernel_size[i]) - 1) - 1) // stride[i] + 1
        for i in range(3)
    )
    effective_lookup_coord_hashmap = st.coord_hashmap if lookup_coord_hashmap is None else lookup_coord_hashmap
    (
        out_rows,
        input_rows_w1,
        input_rows_w2,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        padded_row_count,
        out_coords,
        coord_hashmap,
    ) = _C.gtsparse3d_finalize_row_template_flat_tiled_build_full_runtime_from_coords(
        st.indices.contiguous(),
        int(out_spatial[0]),
        int(out_spatial[1]),
        int(out_spatial[2]),
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        int(max_bm),
        torch.Tensor() if effective_lookup_coord_hashmap is None else effective_lookup_coord_hashmap,
    )
    return FinalizedRowTemplateFlatTiledRuntime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w2=input_rows_w2,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        padded_row_count=padded_row_count,
        n_out=int(out_coords.size(0)),
        bm=int(max_bm),
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=coord_hashmap,
    )


def build_finalize_row_template_flat_tiled_reverse_runtime_from_coords(
    lookup_st: GTSparseSparseConvTensor,
    out_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
    lookup_coord_hashmap: Optional[torch.Tensor] = None,
) -> FinalizedRowTemplateFlatTiledRuntime:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("finalized row-template flat tiled reverse builder currently requires kernel_size=(3, 3, 3)")
    stride = tuple(int(v) for v in stride)
    padding = tuple(int(v) for v in padding)
    dilation = tuple(int(v) for v in dilation)
    out_spatial = tuple(int(v) for v in out_spatial)
    effective_lookup_coord_hashmap = lookup_st.coord_hashmap if lookup_coord_hashmap is None else lookup_coord_hashmap
    (
        out_rows,
        input_rows_w1,
        input_rows_w2,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        padded_row_count,
        coord_hashmap,
    ) = _C.gtsparse3d_finalize_row_template_flat_tiled_build_reverse_runtime_from_coords(
        lookup_st.indices.contiguous(),
        out_coords.contiguous(),
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        int(max_bm),
        torch.Tensor() if effective_lookup_coord_hashmap is None else effective_lookup_coord_hashmap,
    )
    return FinalizedRowTemplateFlatTiledRuntime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w2=input_rows_w2,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        padded_row_count=padded_row_count,
        n_out=int(out_coords.size(0)),
        bm=int(max_bm),
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=coord_hashmap,
    )


def build_finalize_row_template_flat_tiled_reverse_runtime_from_full_runtime(
    forward_runtime: FinalizedRowTemplateFlatTiledRuntime,
    out_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int] | list[int],
    *,
    max_bm: int,
) -> FinalizedRowTemplateFlatTiledRuntime:
    out_spatial = tuple(int(v) for v in out_spatial)
    (
        out_rows,
        input_rows_w1,
        input_rows_w2,
        input_rows_w9,
        input_rows_w18,
        input_rows_w27,
        template_ids,
        input_row_offsets,
        padded_row_count,
    ) = _C.gtsparse3d_finalize_row_template_flat_tiled_build_reverse_runtime_from_full_runtime(
        forward_runtime.out_rows,
        forward_runtime.input_rows_w1,
        forward_runtime.input_rows_w2,
        forward_runtime.input_rows_w9,
        forward_runtime.input_rows_w18,
        forward_runtime.input_rows_w27,
        forward_runtime.template_ids,
        forward_runtime.input_row_offsets,
        forward_runtime.padded_row_count,
        int(out_coords.size(0)),
        int(max_bm),
    )
    return FinalizedRowTemplateFlatTiledRuntime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w2=input_rows_w2,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        padded_row_count=padded_row_count,
        n_out=int(out_coords.size(0)),
        bm=int(max_bm),
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=None,
    )


def finalized_row_template_flat_tiled_subm_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: FinalizedRowTemplateFlatTiledRuntime,
    *,
    config_id: int = 1,
) -> torch.Tensor:
    _require_supported_feature_dtype(features)
    w = _require_flat_weight(features, weight)
    return _finalized_row_template_flat_tiled_forward_unchecked(
        features,
        w,
        runtime,
        config_id=int(config_id),
    )


def finalized_row_template_flat_tiled_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: FinalizedRowTemplateFlatTiledRuntime,
    *,
    config_id: int = 1,
) -> torch.Tensor:
    return finalized_row_template_flat_tiled_subm_conv3d(
        features,
        weight,
        runtime,
        config_id=config_id,
    )


def finalized_row_template_subm_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: FinalizedRowTemplateFlatTiledRuntime,
) -> torch.Tensor:
    _require_supported_feature_dtype(features)
    w = _require_flat_weight(features, weight)
    config_id = _select_finalize_row_template_config_id(
        c_in=int(features.size(1)),
        c_out=int(w.size(1)),
        dtype=features.dtype,
    )
    return _finalized_row_template_flat_tiled_forward_unchecked(
        features,
        w,
        runtime,
        config_id=config_id,
    )


def finalized_row_template_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: FinalizedRowTemplateFlatTiledRuntime,
) -> torch.Tensor:
    return finalized_row_template_subm_conv3d(features, weight, runtime)
