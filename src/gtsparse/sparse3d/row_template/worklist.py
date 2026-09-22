from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch

from gtsparse import _C
from gtsparse.sparse3d.row_template_segmented import (
    SegmentedSubmCodebook as SegmentedCodebookRuntime,
    build_segmented_subm_codebook_from_coords as _build_segmented_subm_codebook_from_coords,
    row_template_segmented_subm_conv3d as _row_template_segmented_subm_conv3d,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor


_NUM_TEMPLATES = 52
_NUM_OFFSETS = 27
_STRIDE_TEMPLATE_END = 39
_LAUNCH_INFO_CACHE: dict[tuple[int, int, int], dict[str, object]] = {}


def _row_template_tile_plan(
    template_counts: torch.Tensor,
    *,
    bm: int,
) -> dict[str, torch.Tensor | int]:
    if template_counts.dtype != torch.int32:
        raise TypeError("template_counts must be int32")
    row_tiles = torch.div(template_counts + int(bm) - 1, int(bm), rounding_mode="floor").to(torch.int32)
    template_regular_tiles = torch.zeros_like(row_tiles)
    template_regular_tiles[: _STRIDE_TEMPLATE_END + 1] = row_tiles[: _STRIDE_TEMPLATE_END + 1]
    regular_tile_prefix = torch.zeros_like(row_tiles)
    if int(row_tiles.numel()) > 1:
        regular_tile_prefix[1:] = torch.cumsum(template_regular_tiles[:-1], dim=0)
    template_tail_tiles = torch.zeros_like(row_tiles)
    template_tail_tiles[_STRIDE_TEMPLATE_END + 1 :] = row_tiles[_STRIDE_TEMPLATE_END + 1 :]
    tail_tile_prefix = torch.zeros_like(row_tiles)
    if int(row_tiles.numel()) > 1:
        tail_tile_prefix[1:] = torch.cumsum(template_tail_tiles[:-1], dim=0)
    payload: dict[str, torch.Tensor | int] = {
        "template_regular_tiles": template_regular_tiles,
        "regular_tile_prefix": regular_tile_prefix,
        "template_tail_tiles": template_tail_tiles,
        "tail_tile_prefix": tail_tile_prefix,
        "num_regular_row_tiles": int(template_regular_tiles.sum().item()),
        "num_tail_row_tiles": int(template_tail_tiles.sum().item()),
    }
    return payload


@dataclass(slots=True)
class RowTemplateWorklist:
    out_in_map: torch.Tensor
    template_rows: torch.Tensor
    template_counts: torch.Tensor
    template_ids: torch.Tensor
    build_max_bm: int
    n_out: int
    _tile_plan_cache: dict[int, dict[str, torch.Tensor | int]] = field(default_factory=dict, init=False, repr=False)

    def tile_plan(self, *, bm: int) -> dict[str, torch.Tensor | int]:
        cached = self._tile_plan_cache.get(int(bm))
        version = int(self.template_counts._version)
        if cached is not None and int(cached["template_counts_version"]) == version:
            return cached
        plan = _row_template_tile_plan(self.template_counts, bm=int(bm))
        plan["template_counts_version"] = version
        self._tile_plan_cache[int(bm)] = plan
        return plan


@dataclass(slots=True)
class CompactRowTemplateRuntime:
    compact_rows: torch.Tensor
    compact_row_ids: torch.Tensor
    compact_out_in_map: torch.Tensor
    template_counts: torch.Tensor
    n_out: int
    bm: int


def row_template_config_table() -> list[dict[str, int | bool | str]]:
    if not hasattr(_C, "gtsparse3d_row_template_subm_conv3d_config_table"):
        return []
    return [{str(k): v for k, v in row.items()} for row in _C.gtsparse3d_row_template_subm_conv3d_config_table()]


def row_template_launch_info(
    *,
    config_id: int = 17,
    c_in: int = 16,
    c_out: int = 64,
) -> dict[str, int | bool | str]:
    if not hasattr(_C, "gtsparse3d_row_template_subm_conv3d_launch_info"):
        return {
            "enabled": False,
            "supported": False,
            "config_id": int(config_id),
            "bm": 0,
            "bn": 0,
            "bk": 0,
            "tm": 0,
            "tn": 0,
            "threads_per_block": 0,
            "shared_storage_bytes": 0,
            "blocks_per_sm": 0,
            "num_sms": 0,
            "grid_dim_x": 0,
            "panel_kind": "row_template",
            "execution_path": "simt_fp16",
            "name": "gtsparse_row_template_subm_conv3d",
        }
    return {
        str(k): v
        for k, v in _C.gtsparse3d_row_template_subm_conv3d_launch_info(int(config_id), int(c_in), int(c_out)).items()
    }


def build_row_template_worklist_from_out_in_map(
    out_in_map: torch.Tensor,
    *,
    max_bm: int,
) -> RowTemplateWorklist:
    if out_in_map.dtype != torch.int32:
        raise TypeError("out_in_map must be int32")
    if out_in_map.dim() != 2 or int(out_in_map.size(1)) != _NUM_OFFSETS:
        raise ValueError("out_in_map must be [N_out, 27]")
    template_rows, template_counts, template_ids = _C.gtsparse3d_build_row_template_from_out_in_map(
        out_in_map.contiguous(),
        int(max_bm),
    )
    return RowTemplateWorklist(
        out_in_map=out_in_map,
        template_rows=template_rows,
        template_counts=template_counts,
        template_ids=template_ids,
        build_max_bm=int(max_bm),
        n_out=int(out_in_map.size(0)),
    )


def build_subm_row_template_worklist_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
) -> RowTemplateWorklist:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("row-template prototype currently requires kernel_size=(3, 3, 3)")
    if tuple(int(v) for v in padding) != (1, 1, 1):
        raise ValueError("row-template prototype currently requires padding=(1, 1, 1)")
    if tuple(int(v) for v in dilation) != (1, 1, 1):
        raise ValueError("row-template fused builder currently requires dilation=(1, 1, 1)")
    out_in_map, template_rows, template_counts, template_ids = _C.gtsparse3d_build_subm_row_template_from_coords(
        st.indices.contiguous(),
        int(max_bm),
    )
    return RowTemplateWorklist(
        out_in_map=out_in_map,
        template_rows=template_rows,
        template_counts=template_counts,
        template_ids=template_ids,
        build_max_bm=int(max_bm),
        n_out=int(out_in_map.size(0)),
    )


def build_subm_row_template_segmented_codebook_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
) -> SegmentedCodebookRuntime:
    return _build_segmented_subm_codebook_from_coords(
        st,
        kernel_size,
        padding,
        dilation,
        max_bm=int(max_bm),
    )


def row_template_subm_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: RowTemplateWorklist,
    *,
    config_id: int = 17,
    tail_mode: str = "ticket",
) -> torch.Tensor:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("row-template kernel currently requires float16 features and weight")
    if not weight.is_contiguous(memory_format=torch.channels_last_3d):
        weight = weight.contiguous(memory_format=torch.channels_last_3d)
    key = (int(config_id), int(features.size(1)), int(weight.size(0)))
    launch = _LAUNCH_INFO_CACHE.get(key)
    if launch is None:
        launch = row_template_launch_info(
            config_id=key[0],
            c_in=key[1],
            c_out=key[2],
        )
        _LAUNCH_INFO_CACHE[key] = launch
    bm = int(launch.get("bm", 0))
    if bm <= 0:
        raise RuntimeError(f"row-template config {int(config_id)} is unavailable")
    if int(runtime.template_rows.size(1)) % bm != 0:
        raise ValueError(
            f"runtime was built with padded width {int(runtime.template_rows.size(1))}, "
            f"which is not divisible by config BM={bm}; rebuild with max_bm multiple of BM"
        )
    tail_mode_id = 0 if tail_mode == "ticket" else 1
    return _C.gtsparse3d_row_template_subm_conv3d_forward(
        features,
        weight,
        runtime.out_in_map,
        runtime.template_rows,
        runtime.template_counts,
        int(config_id),
        tail_mode_id,
    )


def row_template_subm_build_and_conv(
    st: GTSparseSparseConvTensor,
    weight: torch.Tensor,
    *,
    max_bm: int = 128,
    config_id: int = 17,
    tail_mode: str = "ticket",
) -> torch.Tensor:
    if st.features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("row-template kernel currently requires float16 features and weight")
    if not weight.is_contiguous(memory_format=torch.channels_last_3d):
        weight = weight.contiguous(memory_format=torch.channels_last_3d)
    tail_mode_id = 0 if tail_mode == "ticket" else 1
    return _C.gtsparse3d_row_template_subm_build_and_conv(
        st.indices.contiguous(),
        st.features,
        weight,
        int(max_bm),
        int(config_id),
        tail_mode_id,
    )


def row_template_subm_conv3d_cuda_tile_plan(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: RowTemplateWorklist,
    *,
    config_id: int = 17,
    tail_mode: str = "ticket",
    tile_plan: dict[str, torch.Tensor | int] | None = None,
) -> torch.Tensor:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("row-template kernel currently requires float16 features and weight")
    if not weight.is_contiguous(memory_format=torch.channels_last_3d):
        weight = weight.contiguous(memory_format=torch.channels_last_3d)
    key = (int(config_id), int(features.size(1)), int(weight.size(0)))
    launch = _LAUNCH_INFO_CACHE.get(key)
    if launch is None:
        launch = row_template_launch_info(config_id=key[0], c_in=key[1], c_out=key[2])
        _LAUNCH_INFO_CACHE[key] = launch
    bm = int(launch.get("bm", 0))
    if bm <= 0:
        raise RuntimeError(f"row-template config {int(config_id)} is unavailable")
    plan = runtime.tile_plan(bm=bm) if tile_plan is None else tile_plan
    template_regular_tiles = plan["template_regular_tiles"]
    regular_tile_prefix = plan["regular_tile_prefix"]
    template_tail_tiles = plan["template_tail_tiles"]
    tail_tile_prefix = plan["tail_tile_prefix"]
    num_regular_row_tiles = int(plan["num_regular_row_tiles"])
    num_tail_row_tiles = int(plan["num_tail_row_tiles"])
    tail_mode_id = 0 if tail_mode == "ticket" else 1
    return _C.gtsparse3d_row_template_subm_conv3d_forward_tile_plan(
        features,
        weight,
        runtime.out_in_map,
        runtime.template_rows,
        runtime.template_counts,
        template_regular_tiles.contiguous(),
        regular_tile_prefix.contiguous(),
        template_tail_tiles.contiguous(),
        tail_tile_prefix.contiguous(),
        int(num_regular_row_tiles),
        int(num_tail_row_tiles),
        int(config_id),
        tail_mode_id,
    )


# ---- Implicit GEMM path ----


def row_template_igemm_config_table() -> list[dict]:
    return _C.gtsparse3d_row_template_igemm_config_table()


def build_compact_row_template_runtime(
    runtime: RowTemplateWorklist,
    *,
    bm: int,
) -> CompactRowTemplateRuntime:
    if runtime.out_in_map.dtype != torch.int32:
        raise TypeError("runtime.out_in_map must be int32")
    if runtime.template_rows.dtype != torch.int32 or runtime.template_counts.dtype != torch.int32:
        raise TypeError("template metadata must be int32")
    device = runtime.out_in_map.device
    padded_counts = torch.div(
        runtime.template_counts + int(bm) - 1,
        int(bm),
        rounding_mode="floor",
    ).to(torch.int32) * int(bm)
    total_rows = int(padded_counts.sum().item())
    compact_rows = torch.full(
        (total_rows, _NUM_OFFSETS + 2),
        -1,
        dtype=torch.int32,
        device=device,
    )
    compact_row_ids = torch.full((total_rows,), -1, dtype=torch.int32, device=device)
    compact_out_in_map = torch.full((total_rows, _NUM_OFFSETS), -1, dtype=torch.int32, device=device)
    write_row = 0
    for template_id in range(_NUM_TEMPLATES):
        count = int(runtime.template_counts[template_id].item())
        padded = int(padded_counts[template_id].item())
        if padded == 0:
            continue
        if count > 0:
            out_rows = runtime.template_rows[template_id, :count].to(torch.long)
            compact_rows[write_row:write_row + count, 0] = out_rows.to(torch.int32)
            compact_rows[write_row:write_row + count, 1] = int(template_id)
            gathered = runtime.out_in_map.index_select(0, out_rows)
            compact_rows[write_row:write_row + count, 2:] = gathered
            compact_row_ids[write_row:write_row + count] = out_rows.to(torch.int32)
            compact_out_in_map[write_row:write_row + count] = gathered
        if padded > count:
            compact_rows[write_row + count:write_row + padded, 1] = int(template_id)
        write_row += padded
    return CompactRowTemplateRuntime(
        compact_rows=compact_rows,
        compact_row_ids=compact_row_ids,
        compact_out_in_map=compact_out_in_map,
        template_counts=runtime.template_counts,
        n_out=runtime.n_out,
        bm=int(bm),
    )


def row_template_subm_conv3d_igemm(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: RowTemplateWorklist,
    *,
    config_id: int = 0,
    tail_mode: str = "ticket",
) -> torch.Tensor:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("igemm kernel requires float16 features and weight")
    # Weight layout: [kernel_volume, C_in, C_out] row-major
    # Reshape from channels_last_3d [C_out, kD, kH, kW, C_in] if needed
    if weight.dim() == 5:
        w = weight.contiguous(memory_format=torch.channels_last_3d)
        Cout, kD, kH, kW, Cin = w.shape
        w = w.permute(1, 2, 3, 4, 0).contiguous().view(kD * kH * kW * Cin, Cout)
    elif weight.dim() == 3:
        kv, Cin, Cout = weight.shape
        w = weight.contiguous().view(kv * Cin, Cout)
    elif weight.dim() == 2:
        w = weight.contiguous()
    else:
        raise ValueError(f"unexpected weight shape: {weight.shape}")
    return _C.gtsparse3d_row_template_igemm_forward(
        features,
        w,
        runtime.out_in_map,
        runtime.template_rows,
        runtime.template_counts,
        int(runtime.n_out),
        int(config_id),
        tail_mode,
    )


def row_template_subm_conv3d_compact_igemm(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: CompactRowTemplateRuntime,
    *,
    config_id: int = 0,
) -> torch.Tensor:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("compact igemm kernel requires float16 features and weight")
    if weight.dim() == 5:
        w = weight.contiguous(memory_format=torch.channels_last_3d)
        cout, kd, kh, kw, cin = w.shape
        w = w.permute(1, 2, 3, 4, 0).contiguous().view(kd * kh * kw * cin, cout)
    elif weight.dim() == 3:
        kv, cin, cout = weight.shape
        w = weight.contiguous().view(kv * cin, cout)
    elif weight.dim() == 2:
        w = weight.contiguous()
    else:
        raise ValueError(f"unexpected weight shape: {weight.shape}")

    if runtime.compact_rows.numel() == 0:
        return torch.zeros((runtime.n_out, w.size(1)), device=features.device, dtype=features.dtype)

    return _C.gtsparse3d_row_template_compact_igemm_forward(
        features,
        w,
        runtime.compact_rows.contiguous(),
        runtime.compact_row_ids.contiguous(),
        runtime.compact_out_in_map.contiguous(),
        int(runtime.n_out),
        int(config_id),
    )


def row_template_subm_conv3d_persistent_direct_igemm(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: SegmentedCodebookRuntime,
    *,
    config_id: int = 0,
    tail_mode: str = "ticket",
) -> torch.Tensor:
    return _row_template_segmented_subm_conv3d(
        features,
        weight,
        runtime,
        config_id=int(config_id),
        tail_mode=tail_mode,
    )


def row_template_igemm_build_and_conv(
    st: GTSparseSparseConvTensor,
    weight: torch.Tensor,
    *,
    max_bm: int = 128,
    config_id: int = 0,
    tail_mode: str = "ticket",
) -> torch.Tensor:
    if st.features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("igemm kernel requires float16 features and weight")
    if not weight.is_contiguous(memory_format=torch.channels_last_3d):
        weight = weight.contiguous(memory_format=torch.channels_last_3d)
    return _C.gtsparse3d_row_template_igemm_build_and_conv(
        st.indices.contiguous(),
        st.features,
        weight,
        int(max_bm),
        int(config_id),
        tail_mode,
    )
