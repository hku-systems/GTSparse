"""Finalized flattened bundle-worklist builder and migration wrappers.

The finalized operator ABI is the lightweight path described in ``plan.md``:
hash-map/rowmap preprocessing builds one flattened bundle-major worklist where
each item stores one output row plus ``bundle_size`` bundle-local input slots.

The builder itself is already a CUDA path. The convolution wrappers are still
temporary migration adapters and do not yet consume this native flattened ABI
directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from gtsparse import _C
from gtsparse.sparse3d.ew_autotune import get_config_id
from gtsparse.sparse3d.finalize.autotune import (
    decode_finalize_fp16_config_id,
    encode_finalize_fp16_config_id,
    get_finalize_fp16_config_id,
)

from ..expanded_worklist import (
    _active_ratio_from_rowmap,
    _rowmap_to_offset_major_pairs,
    build_full_rowmap_from_coords,
    build_inverse_rowmap_from_coords,
    build_inverse_rowmap_from_coords_into,
    build_subm_rowmap_from_coords,
    build_subm_rowmap_from_coords_into,
)
from ..sparse_tensor import GTSparseSparseConvTensor


_MAX_FINALIZED_OFFSETS = 27
_MAX_CUDA_LOCAL_MASK_SORT_BUNDLE_SIZE = 8
_MAX_FAST_LOCAL_SORT_BUNDLE_SIZE = 5
_FAST_LOCAL_SORT_CHUNK_ITEMS = 256
_DENSE_GRID_BRICK_SIZE = 4
_DENSE_INDEX_GRID_WORKSPACE_CACHE: dict[int, dict[str, torch.Tensor | int]] = {}


def _validate_local_mask_sort_impl(local_mask_sort_impl: str) -> str:
    resolved = str(local_mask_sort_impl)
    if resolved not in {"auto", "cuda", "python"}:
        raise ValueError("local_mask_sort_impl must be one of: auto, cuda, python")
    return resolved


def _local_mask_sort_num_bins(bundle_size: int) -> int:
    if bundle_size in (2, 4):
        return 1 << bundle_size
    return (bundle_size + 1) << bundle_size


def _sort_bundle_segments_by_local_mask_python_(runtime: "CompactTiledWorklist") -> None:
    if runtime.bundle_size <= 1:
        return
    device = runtime.worklist_items.device
    bit_base = 1 << runtime.bundle_size
    for bundle_id in range((int(runtime.offset_counts.numel()) + runtime.bundle_size - 1) // runtime.bundle_size):
        base = bundle_id * runtime.bundle_size
        active_rows = runtime.row_inputs[:, base: base + runtime.bundle_size].ge(0).any(dim=1)
        real_count = int(active_rows.sum().item())
        if real_count <= 1:
            continue
        padded_count = ((real_count + runtime.max_bm - 1) // runtime.max_bm) * runtime.max_bm
        start = 0
        for prev_bundle in range(bundle_id):
            prev_base = prev_bundle * runtime.bundle_size
            prev_real = int(runtime.row_inputs[:, prev_base: prev_base + runtime.bundle_size].ge(0).any(dim=1).sum().item())
            start += ((prev_real + runtime.max_bm - 1) // runtime.max_bm) * runtime.max_bm
        segment = runtime.worklist_items[start: start + real_count]
        local_active = segment[:, 1:1 + runtime.bundle_size].ge(0).to(torch.int32)
        mask = (local_active * (1 << torch.arange(runtime.bundle_size, device=device, dtype=torch.int32))).sum(dim=1)
        if runtime.bundle_size in (2, 4):
            sort_key = mask
        else:
            popcount = local_active.sum(dim=1, dtype=torch.int32)
            sort_key = popcount * bit_base + mask
        order = torch.argsort(sort_key, descending=True)
        runtime.worklist_items[start: start + real_count] = segment.index_select(0, order)


def _sort_bundle_segments_by_local_mask_(
    runtime: "CompactTiledWorklist",
    *,
    local_mask_sort_impl: str = "auto",
    bundle_starts: torch.Tensor | None = None,
    bundle_counts: torch.Tensor | None = None,
    num_items_valid: torch.Tensor | None = None,
    sort_scratch_items: torch.Tensor | None = None,
    sort_keys_in: torch.Tensor | None = None,
    sort_keys_out: torch.Tensor | None = None,
    sort_indices_in: torch.Tensor | None = None,
    sort_indices_out: torch.Tensor | None = None,
    sort_segment_ends: torch.Tensor | None = None,
    fast_sort_chunk_starts: torch.Tensor | None = None,
    fast_sort_chunk_lengths: torch.Tensor | None = None,
    fast_sort_segment_chunk_begin: torch.Tensor | None = None,
    fast_sort_total_chunks_valid: torch.Tensor | None = None,
    fast_sort_chunk_bin_offsets: torch.Tensor | None = None,
) -> None:
    if runtime.bundle_size <= 1:
        return
    resolved_impl = _validate_local_mask_sort_impl(local_mask_sort_impl)
    can_use_cuda_sort = (
        runtime.worklist_items.is_cuda
        and runtime.bundle_size <= _MAX_CUDA_LOCAL_MASK_SORT_BUNDLE_SIZE
        and bundle_starts is not None
        and num_items_valid is not None
        and sort_scratch_items is not None
        and sort_keys_in is not None
        and sort_keys_out is not None
        and sort_indices_in is not None
        and sort_indices_out is not None
        and sort_segment_ends is not None
        and hasattr(_C, "gtsparse3d_sort_compact_worklist_by_local_mask_")
    )
    if resolved_impl in {"auto", "cuda"} and can_use_cuda_sort:
        empty_vec = torch.empty((0,), device=runtime.worklist_items.device, dtype=torch.int32)
        empty_mat = torch.empty((0, 0), device=runtime.worklist_items.device, dtype=torch.int32)
        _C.gtsparse3d_sort_compact_worklist_by_local_mask_(
            runtime.worklist_items,
            bundle_starts,
            num_items_valid,
            runtime.bundle_size,
            sort_scratch_items,
            sort_keys_in,
            sort_keys_out,
            sort_indices_in,
            sort_indices_out,
            empty_vec if bundle_counts is None else bundle_counts,
            sort_segment_ends,
            empty_vec if fast_sort_chunk_starts is None else fast_sort_chunk_starts,
            empty_vec if fast_sort_chunk_lengths is None else fast_sort_chunk_lengths,
            empty_vec if fast_sort_segment_chunk_begin is None else fast_sort_segment_chunk_begin,
            empty_vec if fast_sort_total_chunks_valid is None else fast_sort_total_chunks_valid,
            empty_mat if fast_sort_chunk_bin_offsets is None else fast_sort_chunk_bin_offsets,
        )
        return
    if resolved_impl == "cuda":
        raise ValueError(
            "CUDA local-mask sort requires bundle_size <= 8 and caller-provided builder sort scratch tensors"
        )
    _sort_bundle_segments_by_local_mask_python_(runtime)


def _materialize_cuda_sort_keys_for_runtime(
    runtime: "CompactTiledWorklist",
    *,
    sort_keys_in: torch.Tensor,
    sort_indices_in: torch.Tensor,
    bundle_starts: torch.Tensor | None = None,
    bundle_counts: torch.Tensor | None = None,
) -> None:
    key_limit = _local_mask_sort_num_bins(runtime.bundle_size)
    if bundle_counts is not None and runtime.fixed_bundle_capacity is not None:
        if (
            bundle_starts is not None
            and runtime.worklist_items.is_cuda
            and hasattr(_C, "gtsparse3d_materialize_compact_worklist_local_mask_keys_")
        ):
            _C.gtsparse3d_materialize_compact_worklist_local_mask_keys_(
                runtime.worklist_items,
                bundle_starts,
                bundle_counts,
                runtime.bundle_size,
                sort_keys_in,
                sort_indices_in,
            )
            return
        counts = [int(v) for v in bundle_counts.cpu().tolist()]
        if bundle_starts is None:
            bundle_start_list = [bundle_id * int(runtime.fixed_bundle_capacity) for bundle_id in range(len(counts))]
        else:
            bundle_start_list = [int(v) for v in bundle_starts.cpu().tolist()[: len(counts)]]
        bit_weights = 1 << torch.arange(runtime.bundle_size, device=runtime.worklist_items.device, dtype=torch.int32)
        for bundle_id, count in enumerate(counts):
            if count <= 0:
                continue
            start = bundle_start_list[bundle_id]
            items = runtime.worklist_items[start : start + count]
            local = items[:, 1 : 1 + runtime.bundle_size].ge(0).to(torch.int32)
            mask = (local * bit_weights).sum(dim=1)
            if runtime.bundle_size in (2, 4):
                local_key = mask
            else:
                local_key = local.sum(dim=1, dtype=torch.int32) * (1 << runtime.bundle_size) + mask
            sort_keys_in[start : start + count] = (key_limit - 1) - local_key
            sort_indices_in[start : start + count] = torch.arange(
                start,
                start + count,
                device=items.device,
                dtype=torch.int32,
            )
        return

    total_valid = int(runtime.num_items_valid.item())
    if total_valid <= 0:
        return
    items = runtime.worklist_items[:total_valid]
    local = items[:, 1:1 + runtime.bundle_size].ge(0).to(torch.int32)
    mask = (local * (1 << torch.arange(runtime.bundle_size, device=items.device, dtype=torch.int32))).sum(dim=1)
    if runtime.bundle_size in (2, 4):
        local_key = mask
    else:
        local_key = local.sum(dim=1, dtype=torch.int32) * (1 << runtime.bundle_size) + mask
    sort_keys_in[:total_valid] = (key_limit - 1) - local_key
    sort_indices_in[:total_valid] = torch.arange(total_valid, device=items.device, dtype=torch.int32)


def _get_fixed_capacity_local_mask_sort_workspace(
    *,
    device: torch.device,
    bundle_size: int,
    bundle_capacity: int,
    num_bundles: int,
) -> dict[str, torch.Tensor]:
    total_capacity = int(bundle_capacity) * int(num_bundles)
    item_cols = 1 + int(bundle_size)
    workspace = {
        "bundle_starts": torch.arange(
            0,
            total_capacity,
            int(bundle_capacity),
            device=device,
            dtype=torch.int32,
        ),
        "sort_scratch_items": torch.empty((total_capacity, item_cols), device=device, dtype=torch.int32),
        "sort_keys_in": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_keys_out": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_indices_in": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_indices_out": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_segment_ends": torch.empty((num_bundles,), device=device, dtype=torch.int32),
    }
    if bundle_size <= _MAX_FAST_LOCAL_SORT_BUNDLE_SIZE:
        max_chunks = num_bundles * ((bundle_capacity + _FAST_LOCAL_SORT_CHUNK_ITEMS - 1) // _FAST_LOCAL_SORT_CHUNK_ITEMS)
        num_fast_bins = _local_mask_sort_num_bins(bundle_size) + 1
        workspace["fast_sort_chunk_starts"] = torch.empty((max_chunks,), device=device, dtype=torch.int32)
        workspace["fast_sort_chunk_lengths"] = torch.empty((max_chunks,), device=device, dtype=torch.int32)
        workspace["fast_sort_segment_chunk_begin"] = torch.empty((num_bundles + 1,), device=device, dtype=torch.int32)
        workspace["fast_sort_total_chunks_valid"] = torch.empty((1,), device=device, dtype=torch.int32)
        workspace["fast_sort_chunk_bin_offsets"] = torch.empty((max_chunks, num_fast_bins), device=device, dtype=torch.int32)
    return workspace


def _sort_fixed_capacity_runtime_by_local_mask_(
    runtime: "CompactTiledWorklist",
    *,
    local_mask_sort_impl: str,
) -> None:
    if runtime.bundle_size <= 1:
        return
    bundle_capacity = runtime.fixed_bundle_capacity
    if bundle_capacity is None:
        raise AssertionError("fixed-capacity local-mask sort requires runtime.fixed_bundle_capacity")
    num_bundles = int(runtime.bundle_counts.numel())
    workspace = _get_fixed_capacity_local_mask_sort_workspace(
        device=runtime.worklist_items.device,
        bundle_size=int(runtime.bundle_size),
        bundle_capacity=int(bundle_capacity),
        num_bundles=num_bundles,
    )
    _materialize_cuda_sort_keys_for_runtime(
        runtime,
        sort_keys_in=workspace["sort_keys_in"],
        sort_indices_in=workspace["sort_indices_in"],
        bundle_starts=workspace["bundle_starts"],
        bundle_counts=runtime.bundle_counts,
    )
    _sort_bundle_segments_by_local_mask_(
        runtime,
        local_mask_sort_impl=local_mask_sort_impl,
        bundle_starts=workspace["bundle_starts"],
        bundle_counts=runtime.bundle_counts,
        num_items_valid=runtime.num_items_valid,
        sort_scratch_items=workspace["sort_scratch_items"],
        sort_keys_in=workspace["sort_keys_in"],
        sort_keys_out=workspace["sort_keys_out"],
        sort_indices_in=workspace["sort_indices_in"],
        sort_indices_out=workspace["sort_indices_out"],
        sort_segment_ends=workspace["sort_segment_ends"],
        fast_sort_chunk_starts=workspace.get("fast_sort_chunk_starts"),
        fast_sort_chunk_lengths=workspace.get("fast_sort_chunk_lengths"),
        fast_sort_segment_chunk_begin=workspace.get("fast_sort_segment_chunk_begin"),
        fast_sort_total_chunks_valid=workspace.get("fast_sort_total_chunks_valid"),
        fast_sort_chunk_bin_offsets=workspace.get("fast_sort_chunk_bin_offsets"),
    )
    runtime.worklist_items = workspace["sort_scratch_items"]
    runtime.local_mask_sorted = True


def _get_rowmap_workspace(*, device: torch.device, n_out: int, k_vol: int) -> dict[str, torch.Tensor]:
    return {
        "row_inputs": torch.empty((n_out, k_vol), device=device, dtype=torch.int32),
        "row_masks": torch.empty((n_out,), device=device, dtype=torch.int32),
        "offset_counts": torch.empty((k_vol,), device=device, dtype=torch.int32),
    }


def _build_subm_rowmap_cached(
    coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(_C, "gtsparse3d_build_subm_rowmap_into"):
        return build_subm_rowmap_from_coords(coords, kernel_size, padding, dilation)
    k_vol = int(kernel_size[0] * kernel_size[1] * kernel_size[2])
    workspace = _get_rowmap_workspace(device=coords.device, n_out=int(coords.size(0)), k_vol=k_vol)
    return build_subm_rowmap_from_coords_into(
        coords,
        kernel_size,
        padding,
        dilation,
        workspace["row_inputs"],
        workspace["row_masks"],
        workspace["offset_counts"],
    )


def _build_full_rowmap_cached(
    coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    if not hasattr(_C, "gtsparse3d_build_full_rowmap_into"):
        return build_full_rowmap_from_coords(coords, spatial_shape, kernel_size, stride, padding, dilation)
    out_spatial = [
        (int(spatial_shape[i]) + 2 * int(padding[i]) - int(dilation[i]) * (int(kernel_size[i]) - 1) - 1)
        // int(stride[i]) + 1
        for i in range(3)
    ]
    out_coords = _C.gtsparse3d_enumerate_output_coords(
        coords,
        out_spatial[0], out_spatial[1], out_spatial[2],
        int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2]),
        int(stride[0]), int(stride[1]), int(stride[2]),
        int(padding[0]), int(padding[1]), int(padding[2]),
        int(dilation[0]), int(dilation[1]), int(dilation[2]),
    )
    k_vol = int(kernel_size[0] * kernel_size[1] * kernel_size[2])
    workspace = _get_rowmap_workspace(device=coords.device, n_out=int(out_coords.size(0)), k_vol=k_vol)
    row_inputs, row_masks, offset_counts, global_offset_support = _C.gtsparse3d_build_full_rowmap_into(
        out_coords,
        coords,
        int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2]),
        int(stride[0]), int(stride[1]), int(stride[2]),
        int(padding[0]), int(padding[1]), int(padding[2]),
        int(dilation[0]), int(dilation[1]), int(dilation[2]),
        workspace["row_inputs"],
        workspace["row_masks"],
        workspace["offset_counts"],
    )
    return row_inputs, row_masks, offset_counts, global_offset_support, out_coords, out_spatial


def _build_inverse_rowmap_cached(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(_C, "gtsparse3d_build_inverse_rowmap_into"):
        return build_inverse_rowmap_from_coords(in_coords, out_coords, kernel_size, stride, padding, dilation)
    k_vol = int(kernel_size[0] * kernel_size[1] * kernel_size[2])
    workspace = _get_rowmap_workspace(device=out_coords.device, n_out=int(out_coords.size(0)), k_vol=k_vol)
    return build_inverse_rowmap_from_coords_into(
        in_coords,
        out_coords,
        kernel_size,
        stride,
        padding,
        dilation,
        workspace["row_inputs"],
        workspace["row_masks"],
        workspace["offset_counts"],
    )


def _get_soa_workspace(*, device: torch.device, n_out: int, max_bm: int, bundle_size: int) -> dict[str, torch.Tensor]:
    bundle_capacity = ((n_out + max_bm - 1) // max_bm) * max_bm
    num_bundles = (_MAX_FINALIZED_OFFSETS + bundle_size - 1) // bundle_size
    item_cols = 1 + bundle_size
    bundle_slots = num_bundles * bundle_size
    total_capacity = num_bundles * bundle_capacity
    workspace = {
        "worklist_items": torch.empty((total_capacity, item_cols), device=device, dtype=torch.int32),
        "num_items_valid": torch.zeros((1,), device=device, dtype=torch.int32),
        "row_masks": torch.empty((n_out,), device=device, dtype=torch.int32),
        "offset_counts": torch.empty((27,), device=device, dtype=torch.int32),
        "bundle_counts": torch.empty((num_bundles,), device=device, dtype=torch.int32),
        "bundle_starts": torch.empty((num_bundles,), device=device, dtype=torch.int32),
        "bundle_write_ptrs": torch.empty((num_bundles,), device=device, dtype=torch.int32),
        "bundle_inputs": torch.empty((bundle_slots, n_out), device=device, dtype=torch.int32),
        "sort_scratch_items": torch.empty((total_capacity, item_cols), device=device, dtype=torch.int32),
        "sort_keys_in": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_keys_out": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_indices_in": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_indices_out": torch.empty((total_capacity,), device=device, dtype=torch.int32),
        "sort_segment_ends": torch.empty((num_bundles,), device=device, dtype=torch.int32),
        "fast_sort_chunk_starts": torch.empty((num_bundles * ((bundle_capacity + _FAST_LOCAL_SORT_CHUNK_ITEMS - 1) // _FAST_LOCAL_SORT_CHUNK_ITEMS),), device=device, dtype=torch.int32),
        "fast_sort_chunk_lengths": torch.empty((num_bundles * ((bundle_capacity + _FAST_LOCAL_SORT_CHUNK_ITEMS - 1) // _FAST_LOCAL_SORT_CHUNK_ITEMS),), device=device, dtype=torch.int32),
        "fast_sort_segment_chunk_begin": torch.empty((num_bundles + 1,), device=device, dtype=torch.int32),
        "fast_sort_total_chunks_valid": torch.empty((1,), device=device, dtype=torch.int32),
        "fast_sort_chunk_bin_offsets": torch.empty((num_bundles * ((bundle_capacity + _FAST_LOCAL_SORT_CHUNK_ITEMS - 1) // _FAST_LOCAL_SORT_CHUNK_ITEMS), 17), device=device, dtype=torch.int32),
    }
    return workspace


def _get_dense_index_grid_workspace(
    *,
    device: torch.device,
    batch_size: int,
    spatial_shape: Tuple[int, int, int] | list[int],
) -> dict[str, torch.Tensor]:
    d, h, w = (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
    key = int(device.index or 0)
    cached = _DENSE_INDEX_GRID_WORKSPACE_CACHE.get(key)
    if cached is not None:
        required_slots = int(batch_size) * ((d + _DENSE_GRID_BRICK_SIZE - 1) // _DENSE_GRID_BRICK_SIZE) * ((h + _DENSE_GRID_BRICK_SIZE - 1) // _DENSE_GRID_BRICK_SIZE) * ((w + _DENSE_GRID_BRICK_SIZE - 1) // _DENSE_GRID_BRICK_SIZE) * (_DENSE_GRID_BRICK_SIZE ** 3)
        if int(cached["index_grid"].numel()) >= required_slots:
            return cached
    tile_d = (d + _DENSE_GRID_BRICK_SIZE - 1) // _DENSE_GRID_BRICK_SIZE
    tile_h = (h + _DENSE_GRID_BRICK_SIZE - 1) // _DENSE_GRID_BRICK_SIZE
    tile_w = (w + _DENSE_GRID_BRICK_SIZE - 1) // _DENSE_GRID_BRICK_SIZE
    grid_slots = int(batch_size) * tile_d * tile_h * tile_w * (_DENSE_GRID_BRICK_SIZE ** 3)
    workspace = {
        "index_grid": torch.zeros((grid_slots,), device=device, dtype=torch.int32),
        "epoch": 0,
    }
    _DENSE_INDEX_GRID_WORKSPACE_CACHE[key] = workspace
    return workspace


def _brick_row_reorder_perm(
    coords: torch.Tensor,
    *,
    spatial_shape: Tuple[int, int, int] | list[int],
    brick_size: int = _DENSE_GRID_BRICK_SIZE,
) -> torch.Tensor | None:
    n_rows = int(coords.size(0))
    if n_rows <= 1:
        return None
    d, h, w = (int(spatial_shape[0]), int(spatial_shape[1]), int(spatial_shape[2]))
    tiles_d = (d + brick_size - 1) // brick_size
    tiles_h = (h + brick_size - 1) // brick_size
    tiles_w = (w + brick_size - 1) // brick_size
    batch = coords[:, 0].to(torch.int64)
    brick_d = torch.div(coords[:, 1], brick_size, rounding_mode="floor").to(torch.int64)
    brick_h = torch.div(coords[:, 2], brick_size, rounding_mode="floor").to(torch.int64)
    brick_w = torch.div(coords[:, 3], brick_size, rounding_mode="floor").to(torch.int64)
    brick_id = ((batch * tiles_d + brick_d) * tiles_h + brick_h) * tiles_w + brick_w
    perm = torch.argsort(brick_id)
    identity = torch.arange(n_rows, device=coords.device, dtype=perm.dtype)
    if bool(torch.equal(perm, identity)):
        return None
    return perm


def _remap_worklist_items_by_perm_(runtime: "CompactTiledWorklist", perm: torch.Tensor) -> None:
    valid = int(runtime.num_items_valid.item())
    if valid <= 0:
        return
    items = runtime.worklist_items[:valid]
    mask = items.ge(0)
    if not bool(mask.any()):
        return
    mapped = perm.index_select(0, items[mask].to(torch.int64)).to(torch.int32)
    items[mask] = mapped


def _resolve_finalize_execution_path(
    dtype: torch.dtype,
    allow_tf32: bool | None = None,
) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype != torch.float32:
        raise TypeError(f"finalized baseline query does not support dtype={dtype}")
    resolved_allow_tf32 = torch.backends.cuda.matmul.allow_tf32 if allow_tf32 is None else bool(allow_tf32)
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    sm = capability[0] * 10 + capability[1]
    return "tf32" if resolved_allow_tf32 and sm >= 80 else "simt"


def _finalized_query_symbol(panel_kind: str, execution_path: str, kind: str) -> str:
    if panel_kind not in {"naive", "sticky"}:
        raise ValueError("panel_kind must be one of: naive, sticky")
    if execution_path not in {"simt", "tf32", "fp16"}:
        raise ValueError("execution_path must be one of: simt, tf32, fp16")
    if kind not in {"config_table", "launch_info"}:
        raise ValueError("kind must be one of: config_table, launch_info")
    if panel_kind == "naive":
        prefix = "gtsparse3d_finalize_naive_worklist_conv3d"
    else:
        prefix = "gtsparse3d_finalize_sticky_offset_conv3d"
    return f"{prefix}_{execution_path}_{kind}"


def _disabled_finalize_launch_info(panel_kind: str, execution_path: str, config_id: int) -> dict[str, int | bool | str]:
    return {
        "enabled": False,
        "supported": False,
        "config_id": int(config_id),
        "panel_kind": panel_kind,
        "execution_path": execution_path,
        "name": f"gtsparse_finalize_{panel_kind}_{execution_path}",
    }


def finalized_baseline_config_table(
    panel_kind: str,
    *,
    dtype: torch.dtype,
    allow_tf32: bool | None = None,
) -> list[dict[str, int | bool | str]]:
    execution_path = _resolve_finalize_execution_path(dtype, allow_tf32=allow_tf32)
    symbol = _finalized_query_symbol(panel_kind, execution_path, "config_table")
    if not hasattr(_C, symbol):
        return []
    return [{str(k): v for k, v in row.items()} for row in getattr(_C, symbol)()]


def finalized_baseline_launch_info(
    panel_kind: str,
    *,
    dtype: torch.dtype,
    config_id: int,
    c_in: int = 16,
    c_out: int = 128,
    allow_tf32: bool | None = None,
) -> dict[str, int | bool | str]:
    execution_path = _resolve_finalize_execution_path(dtype, allow_tf32=allow_tf32)
    symbol = _finalized_query_symbol(panel_kind, execution_path, "launch_info")
    if not hasattr(_C, symbol):
        return _disabled_finalize_launch_info(panel_kind, execution_path, config_id)
    return {str(k): v for k, v in getattr(_C, symbol)(config_id, c_in, c_out).items()}


def finalized_naive_config_table(
    *,
    dtype: torch.dtype,
    allow_tf32: bool | None = None,
) -> list[dict[str, int | bool | str]]:
    return finalized_baseline_config_table("naive", dtype=dtype, allow_tf32=allow_tf32)


def finalized_naive_launch_info(
    *,
    dtype: torch.dtype,
    config_id: int,
    c_in: int = 16,
    c_out: int = 128,
    allow_tf32: bool | None = None,
) -> dict[str, int | bool | str]:
    return finalized_baseline_launch_info(
        "naive",
        dtype=dtype,
        config_id=config_id,
        c_in=c_in,
        c_out=c_out,
        allow_tf32=allow_tf32,
    )


def finalized_sticky_config_table(
    *,
    dtype: torch.dtype,
    allow_tf32: bool | None = None,
) -> list[dict[str, int | bool | str]]:
    return finalized_baseline_config_table("sticky", dtype=dtype, allow_tf32=allow_tf32)


def finalized_sticky_launch_info(
    *,
    dtype: torch.dtype,
    config_id: int,
    c_in: int = 16,
    c_out: int = 128,
    allow_tf32: bool | None = None,
) -> dict[str, int | bool | str]:
    return finalized_baseline_launch_info(
        "sticky",
        dtype=dtype,
        config_id=config_id,
        c_in=c_in,
        c_out=c_out,
        allow_tf32=allow_tf32,
    )


def finalized_static_bundle_config_table() -> list[dict[str, int | bool | str]]:
    if not hasattr(_C, "gtsparse3d_finalize_static_bundle_conv3d_fp16_config_table"):
        return []
    return [
        {str(k): v for k, v in row.items()}
        for row in _C.gtsparse3d_finalize_static_bundle_conv3d_fp16_config_table()
    ]


def finalized_static_bundle_simt_config_table() -> list[dict[str, int | bool | str]]:
    if not hasattr(_C, "gtsparse3d_finalize_static_bundle_conv3d_simt_config_table"):
        return []
    return [
        {str(k): v for k, v in row.items()}
        for row in _C.gtsparse3d_finalize_static_bundle_conv3d_simt_config_table()
    ]


def finalized_static_bundle_launch_info(
    *,
    config_id: int,
    c_in: int = 16,
    c_out: int = 128,
) -> dict[str, int | bool | str]:
    if not hasattr(_C, "gtsparse3d_finalize_static_bundle_conv3d_fp16_launch_info"):
        return {
            "enabled": False,
            "supported": False,
            "config_id": int(config_id),
            "panel_kind": "bundle",
            "execution_path": "fp16",
            "name": "gtsparse_finalize_static_bundle_fp16",
        }
    return {
        str(k): v
        for k, v in _C.gtsparse3d_finalize_static_bundle_conv3d_fp16_launch_info(config_id, c_in, c_out).items()
    }


def finalized_static_bundle_simt_launch_info(
    *,
    config_id: int,
    c_in: int = 16,
    c_out: int = 128,
) -> dict[str, int | bool | str]:
    if not hasattr(_C, "gtsparse3d_finalize_static_bundle_conv3d_simt_launch_info"):
        return {
            "enabled": False,
            "supported": False,
            "config_id": int(config_id),
            "panel_kind": "bundle",
            "execution_path": "simt",
            "name": "gtsparse_finalize_static_bundle_simt",
        }
    return {
        str(k): v
        for k, v in _C.gtsparse3d_finalize_static_bundle_conv3d_simt_launch_info(config_id, c_in, c_out).items()
    }


def _resolve_finalize_fp16_local_config_id(
    panel_kind: str,
    config_id: int,
    *,
    expected_bundle_size: int | None = None,
) -> int:
    if config_id < 0:
        raise ValueError("config_id must be non-negative before decoding finalized fp16 shared-space ids")
    space_variant, space_bundle_size, local = decode_finalize_fp16_config_id(config_id)
    if space_variant == panel_kind:
        if panel_kind == "bundle" and expected_bundle_size is not None and space_bundle_size != expected_bundle_size:
            raise AssertionError(
                f"finalized fp16 config_id {config_id} belongs to bundle_size={space_bundle_size}, expected {expected_bundle_size}"
            )
        return int(local)
    if panel_kind == "naive" and config_id < len(finalized_naive_config_table(dtype=torch.float16)):
        return int(config_id)
    if panel_kind == "sticky" and config_id < len(finalized_sticky_config_table(dtype=torch.float16)):
        return int(config_id)
    if panel_kind == "bundle" and config_id < len(finalized_static_bundle_config_table()):
        return int(config_id)
    raise AssertionError(f"finalized fp16 config_id {config_id} does not belong to panel_kind={panel_kind}")


def finalized_fp16_worklist_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    runtime: CompactTiledWorklist,
    *,
    config_id: int = -1,
    active_ratio: float | None = None,
    candidate_variants: Tuple[str, ...] | list[str] | None = None,
    candidate_bundle_sizes: Tuple[int, ...] | list[int] | None = None,
    bundle_runtime_map: dict[int, CompactTiledWorklist] | None = None,
) -> tuple[torch.Tensor, str, int, int, int | None]:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("finalized shared autotune entry currently requires float16 features and weight")
    if active_ratio is None:
        active_ratio = _active_ratio_from_rowmap(runtime.offset_counts, runtime.n_out)
    if config_id < 0:
        global_config_id = get_finalize_fp16_config_id(
            features.size(1),
            weight.size(0),
            int(runtime.offset_counts.numel()),
            features,
            weight,
            runtime,
            active_ratio=active_ratio,
            candidate_variants=candidate_variants,
            candidate_bundle_sizes=candidate_bundle_sizes,
            bundle_runtime_map=bundle_runtime_map,
        )
    else:
        global_config_id = int(config_id)
    variant, bundle_size, local_config_id = decode_finalize_fp16_config_id(global_config_id)
    if variant == "naive":
        out = finalized_naive_worklist_conv3d_cuda(features, weight, bias, runtime, config_id=local_config_id)
    elif variant == "sticky":
        out = finalized_sticky_offset_conv3d_cuda(features, weight, bias, runtime, config_id=local_config_id)
    elif variant == "bundle":
        if bundle_runtime_map is None or bundle_size not in bundle_runtime_map:
            raise AssertionError(f"bundle runtime for bundle_size={bundle_size} is required for shared finalized autotune dispatch")
        out = finalized_static_bundle_conv3d_cuda(features, weight, bias, bundle_runtime_map[bundle_size], config_id=local_config_id)
    else:
        raise AssertionError(f"unknown finalized fp16 variant {variant}")
    return out, variant, int(global_config_id), int(local_config_id), bundle_size


@dataclass(slots=True)
class CompactTiledWorklist:
    """Finalized flattened bundle-major worklist contract.

    Fields:
        worklist_items:
            ``[num_items_pad, 1 + bundle_size]`` int32. Each item stores
            ``[output_row, bundle-local input rows...]`` with ``-1`` marking an
            inactive bundle-local offset or a padded row.
        bundle_counts:
            ``[num_bundles]`` int32. Number of real worklist items in each
            bundle segment before per-bundle ``max_bm`` padding.
        num_items_valid:
            Scalar int32 tensor tracking the valid padded prefix length inside
            the worst-case allocated worklist tensors.
        row_inputs / row_masks / offset_counts:
            The lightweight rowmap/hash-map contract from which the compact
            worklist was built. These are kept so the current mainline kernels
            can still be used as migration backends.
        fixed_bundle_capacity:
            Optional fixed per-bundle segment width for direct builders that
            reserve the same padded capacity for every bundle. When ``None``,
            segment starts are reconstructed from ``bundle_counts`` and
            per-bundle ``max_bm`` padding.
        coord_hashmap:
            Optional `[capacity, 2]` int64 coordinate-hash buckets associated
            with the output active set. Full-conv preprocess can hand this to
            the next SubM layer to avoid rebuilding the coordinate hashmap.
    """

    worklist_items: torch.Tensor
    bundle_counts: torch.Tensor
    num_items_valid: torch.Tensor
    row_inputs: torch.Tensor
    row_masks: torch.Tensor
    offset_counts: torch.Tensor
    global_offset_support: torch.Tensor
    max_bm: int
    bundle_size: int
    n_out: int
    fixed_bundle_capacity: int | None = None
    coord_hashmap: torch.Tensor | None = None
    local_mask_sorted: bool = False

    @property
    def num_items_capacity(self) -> int:
        return int(self.worklist_items.size(0))


def _get_compact_worklist_workspace(
    *,
    row_inputs: torch.Tensor,
    max_bm: int,
    bundle_size: int,
) -> dict[str, torch.Tensor]:
    n_out = int(row_inputs.size(0))
    k_vol = int(row_inputs.size(1))
    num_bundles = (k_vol + bundle_size - 1) // bundle_size
    item_cols = 1 + bundle_size
    bundle_capacity = ((n_out + max_bm - 1) // max_bm) * max_bm
    total_capacity = num_bundles * bundle_capacity
    workspace = {
        "worklist_items": torch.empty((total_capacity, item_cols), device=row_inputs.device, dtype=torch.int32),
        "num_items_valid": torch.zeros((1,), device=row_inputs.device, dtype=torch.int32),
        "bundle_counts": torch.empty((num_bundles,), device=row_inputs.device, dtype=torch.int32),
        "bundle_starts": torch.empty((num_bundles,), device=row_inputs.device, dtype=torch.int32),
        "bundle_write_ptrs": torch.empty((num_bundles,), device=row_inputs.device, dtype=torch.int32),
        "sort_scratch_items": torch.empty((total_capacity, item_cols), device=row_inputs.device, dtype=torch.int32),
        "sort_keys_in": torch.empty((total_capacity,), device=row_inputs.device, dtype=torch.int32),
        "sort_keys_out": torch.empty((total_capacity,), device=row_inputs.device, dtype=torch.int32),
        "sort_indices_in": torch.empty((total_capacity,), device=row_inputs.device, dtype=torch.int32),
        "sort_indices_out": torch.empty((total_capacity,), device=row_inputs.device, dtype=torch.int32),
        "sort_segment_ends": torch.empty((num_bundles,), device=row_inputs.device, dtype=torch.int32),
    }
    if bundle_size <= _MAX_FAST_LOCAL_SORT_BUNDLE_SIZE:
        max_chunks = num_bundles * ((bundle_capacity + _FAST_LOCAL_SORT_CHUNK_ITEMS - 1) // _FAST_LOCAL_SORT_CHUNK_ITEMS)
        num_fast_bins = _local_mask_sort_num_bins(bundle_size) + 1
        workspace["fast_sort_chunk_starts"] = torch.empty((max_chunks,), device=row_inputs.device, dtype=torch.int32)
        workspace["fast_sort_chunk_lengths"] = torch.empty((max_chunks,), device=row_inputs.device, dtype=torch.int32)
        workspace["fast_sort_segment_chunk_begin"] = torch.empty((num_bundles + 1,), device=row_inputs.device, dtype=torch.int32)
        workspace["fast_sort_total_chunks_valid"] = torch.empty((1,), device=row_inputs.device, dtype=torch.int32)
        workspace["fast_sort_chunk_bin_offsets"] = torch.empty((max_chunks, num_fast_bins), device=row_inputs.device, dtype=torch.int32)
    return workspace


def _check_rowmap_contract(
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
    bundle_size: int,
) -> None:
    if row_inputs.dtype != torch.int32 or row_masks.dtype != torch.int32 or offset_counts.dtype != torch.int32:
        raise TypeError("row_inputs, row_masks, and offset_counts must all be int32")
    if row_inputs.dim() != 2 or row_masks.dim() != 1 or offset_counts.dim() != 1:
        raise ValueError("row_inputs must be [N_out, K], row_masks [N_out], and offset_counts [K]")
    if row_inputs.size(0) != row_masks.numel():
        raise ValueError("row_inputs.size(0) must match row_masks.numel()")
    if row_inputs.size(1) != offset_counts.numel():
        raise ValueError("row_inputs.size(1) must match offset_counts.numel()")

    k_vol = int(offset_counts.numel())
    if k_vol > _MAX_FINALIZED_OFFSETS:
        raise ValueError(f"finalized compact worklist requires num_offsets <= {_MAX_FINALIZED_OFFSETS}, got {k_vol}")
    if bundle_size <= 0:
        raise ValueError("bundle_size must be positive")


def _empty_compact_tiled_worklist(
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
    global_offset_support: torch.Tensor,
    *,
    max_bm: int,
    bundle_size: int,
) -> CompactTiledWorklist:
    device = row_inputs.device
    return CompactTiledWorklist(
        worklist_items=torch.empty((0, 1 + bundle_size), device=device, dtype=torch.int32),
        bundle_counts=torch.empty((0,), device=device, dtype=torch.int32),
        num_items_valid=torch.zeros((1,), device=device, dtype=torch.int32),
        row_inputs=row_inputs,
        row_masks=row_masks,
        offset_counts=offset_counts,
        global_offset_support=global_offset_support,
        max_bm=int(max_bm),
        bundle_size=int(bundle_size),
        n_out=int(row_inputs.size(0)),
        fixed_bundle_capacity=((int(row_inputs.size(0)) + int(max_bm) - 1) // int(max_bm)) * int(max_bm),
        coord_hashmap=None,
    )


def build_compact_tiled_worklist_from_rowmap(
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
    *,
    max_bm: int,
    bundle_size: int = 1,
    global_offset_support: torch.Tensor | None = None,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> CompactTiledWorklist:
    """Build the flattened finalized bundle-major worklist from rowmap inputs."""
    _check_rowmap_contract(row_inputs, row_masks, offset_counts, bundle_size)
    if max_bm <= 0:
        raise ValueError("max_bm must be positive")
    resolved_local_mask_sort_impl = _validate_local_mask_sort_impl(local_mask_sort_impl)

    if global_offset_support is None:
        global_offset_support = row_inputs.ge(0).sum(dim=0, dtype=torch.int32)
    elif global_offset_support.dtype != torch.int32 or global_offset_support.dim() != 1:
        raise TypeError("global_offset_support must be [K] int32 when provided")

    if row_inputs.size(0) == 0 or offset_counts.numel() == 0:
        return _empty_compact_tiled_worklist(
            row_inputs,
            row_masks,
            offset_counts,
            global_offset_support,
            max_bm=max_bm,
            bundle_size=bundle_size,
        )

    workspace = _get_compact_worklist_workspace(
        row_inputs=row_inputs,
        max_bm=int(max_bm),
        bundle_size=int(bundle_size),
    )

    worklist_items, bundle_counts, num_items_valid = _C.gtsparse3d_build_compact_worklist_from_rowmap_into(
        row_inputs,
        row_masks,
        offset_counts,
        int(max_bm),
        int(bundle_size),
        workspace["worklist_items"],
        workspace["num_items_valid"],
        workspace["bundle_counts"],
        workspace["bundle_starts"],
        workspace["bundle_write_ptrs"],
        workspace["sort_keys_in"],
        workspace["sort_indices_in"],
        bool(sort_by_local_mask and bundle_size > 1),
    )

    runtime = CompactTiledWorklist(
        worklist_items=worklist_items,
        bundle_counts=bundle_counts,
        num_items_valid=num_items_valid,
        row_inputs=row_inputs,
        row_masks=row_masks,
        offset_counts=offset_counts,
        global_offset_support=global_offset_support,
        max_bm=int(max_bm),
        bundle_size=int(bundle_size),
        n_out=int(row_inputs.size(0)),
        fixed_bundle_capacity=((int(row_inputs.size(0)) + int(max_bm) - 1) // int(max_bm)) * int(max_bm),
        local_mask_sorted=False,
    )
    if sort_by_local_mask:
        _sort_bundle_segments_by_local_mask_(
            runtime,
            local_mask_sort_impl=resolved_local_mask_sort_impl,
            bundle_starts=workspace["bundle_starts"],
            num_items_valid=workspace["num_items_valid"],
            sort_scratch_items=workspace["sort_scratch_items"],
            sort_keys_in=workspace["sort_keys_in"],
            sort_keys_out=workspace["sort_keys_out"],
            sort_indices_in=workspace["sort_indices_in"],
            sort_indices_out=workspace["sort_indices_out"],
            sort_segment_ends=workspace["sort_segment_ends"],
            fast_sort_chunk_starts=workspace.get("fast_sort_chunk_starts"),
            fast_sort_chunk_lengths=workspace.get("fast_sort_chunk_lengths"),
            fast_sort_segment_chunk_begin=workspace.get("fast_sort_segment_chunk_begin"),
            fast_sort_total_chunks_valid=workspace.get("fast_sort_total_chunks_valid"),
            fast_sort_chunk_bin_offsets=workspace.get("fast_sort_chunk_bin_offsets"),
        )
        workspace["worklist_items"], workspace["sort_scratch_items"] = (
            workspace["sort_scratch_items"],
            workspace["worklist_items"],
        )
        runtime.worklist_items = workspace["worklist_items"]
        runtime.local_mask_sorted = True
    return runtime


def materialize_legacy_offset_major_pairs(
    runtime: CompactTiledWorklist,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Materialize the old padded contract from finalized rowmap metadata.

    This is a migration helper so the current mainline kernels can execute while
    the finalized compact operators are being moved into their dedicated kernel
    folder.

    This bridge is intentionally limited to the legacy offset-major execution
    shape; it ignores the flattened worklist payload and reconstructs the old
    padded contract from ``row_inputs``.
    """
    pairs = _rowmap_to_offset_major_pairs(runtime.row_inputs, runtime.offset_counts)
    k_vol = int(runtime.offset_counts.numel())
    n_stride = int(pairs.size(0) // max(k_vol, 1)) if k_vol > 0 else 0
    return pairs, runtime.offset_counts, n_stride


def _resolve_naive_config_id(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: CompactTiledWorklist,
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    n_stride: int,
    config_id: int,
    active_ratio: float | None,
) -> int:
    if config_id >= 0:
        return int(config_id)
    k_vol = int(offset_counts.numel())
    if active_ratio is None:
        active_ratio = _active_ratio_from_rowmap(offset_counts, runtime.n_out)
    return int(
        get_config_id(
            features.size(1),
            weight.size(0),
            k_vol,
            features.dtype,
            features,
            weight,
            pairs,
            offset_counts,
            n_stride,
            runtime.n_out,
            active_ratio,
        )
    )


def finalized_naive_worklist_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    runtime: CompactTiledWorklist,
    *,
    config_id: int = -1,
    active_ratio: float | None = None,
    allow_tf32: bool | None = None,
) -> torch.Tensor:
    """Run the clean finalized naive worklist surface.

    This baseline currently supports only the `bundle_size = 1` specialization
    of the flattened finalized ABI.
    """
    if runtime.bundle_size != 1:
        raise AssertionError("finalized naive baseline only supports bundle_size=1")
    if features.dtype != weight.dtype or features.dtype not in (torch.float32, torch.float16):
        raise TypeError("finalized naive baseline currently requires matching float32 or float16 features and weight")

    resolved_allow_tf32 = torch.backends.cuda.matmul.allow_tf32 if allow_tf32 is None else bool(allow_tf32)
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    restore_tf32 = bool(features.dtype == torch.float32 and old_allow_tf32 != resolved_allow_tf32)

    try:
        if restore_tf32:
            torch.backends.cuda.matmul.allow_tf32 = resolved_allow_tf32

        if config_id < 0:
            if features.dtype == torch.float16:
                global_config_id = get_finalize_fp16_config_id(
                    features.size(1),
                    weight.size(0),
                    int(runtime.offset_counts.numel()),
                    features,
                    weight,
                    runtime,
                    active_ratio=active_ratio or _active_ratio_from_rowmap(runtime.offset_counts, runtime.n_out),
                    candidate_variants=("naive",),
                )
                resolved_config_id = _resolve_finalize_fp16_local_config_id("naive", global_config_id)
                return _C.gtsparse3d_finalize_naive_worklist_conv3d_forward(
                    features,
                    weight,
                    bias,
                    runtime.worklist_items,
                    runtime.bundle_counts,
                    runtime.max_bm,
                    int(runtime.fixed_bundle_capacity),
                    runtime.n_out,
                    resolved_config_id,
                    resolved_allow_tf32,
                )
            pairs, offset_counts, n_stride = materialize_legacy_offset_major_pairs(runtime)
        else:
            pairs = torch.empty((0, 2), device=runtime.row_inputs.device, dtype=torch.int32)
            offset_counts = runtime.offset_counts
            n_stride = 0

        resolved_config_id = _resolve_naive_config_id(
            features,
            weight,
            runtime,
            pairs,
            offset_counts,
            n_stride,
            config_id,
            active_ratio,
        )
        return _C.gtsparse3d_finalize_naive_worklist_conv3d_forward(
            features,
            weight,
            bias,
            runtime.worklist_items,
            runtime.bundle_counts,
            runtime.max_bm,
            int(runtime.fixed_bundle_capacity),
            runtime.n_out,
            resolved_config_id,
            resolved_allow_tf32,
        )
    finally:
        if restore_tf32:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


def finalized_sticky_offset_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    runtime: CompactTiledWorklist,
    *,
    config_id: int = -1,
    active_ratio: float | None = None,
    allow_tf32: bool | None = None,
) -> torch.Tensor:
    if runtime.bundle_size != 1:
        raise AssertionError("finalized sticky baseline only supports bundle_size=1")
    if features.dtype != weight.dtype or features.dtype not in (torch.float32, torch.float16):
        raise TypeError("sticky offset finalized path currently requires matching float32 or float16 features and weight")

    resolved_allow_tf32 = torch.backends.cuda.matmul.allow_tf32 if allow_tf32 is None else bool(allow_tf32)
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    restore_tf32 = bool(features.dtype == torch.float32 and old_allow_tf32 != resolved_allow_tf32)

    try:
        if restore_tf32:
            torch.backends.cuda.matmul.allow_tf32 = resolved_allow_tf32

        if config_id < 0:
            if features.dtype == torch.float16:
                global_config_id = get_finalize_fp16_config_id(
                    features.size(1),
                    weight.size(0),
                    int(runtime.offset_counts.numel()),
                    features,
                    weight,
                    runtime,
                    active_ratio=active_ratio or _active_ratio_from_rowmap(runtime.offset_counts, runtime.n_out),
                    candidate_variants=("sticky",),
                )
                resolved_config_id = _resolve_finalize_fp16_local_config_id("sticky", global_config_id)
                return _C.gtsparse3d_finalize_sticky_offset_conv3d_forward(
                    features,
                    weight,
                    bias,
                    runtime.worklist_items,
                    runtime.bundle_counts,
                    runtime.max_bm,
                    int(runtime.fixed_bundle_capacity),
                    runtime.n_out,
                    resolved_config_id,
                    resolved_allow_tf32,
                )
            pairs, offset_counts, n_stride = materialize_legacy_offset_major_pairs(runtime)
        else:
            pairs = torch.empty((0, 2), device=runtime.row_inputs.device, dtype=torch.int32)
            offset_counts = runtime.offset_counts
            n_stride = 0

        resolved_config_id = _resolve_naive_config_id(
            features,
            weight,
            runtime,
            pairs,
            offset_counts,
            n_stride,
            config_id,
            active_ratio,
        )
        return _C.gtsparse3d_finalize_sticky_offset_conv3d_forward(
            features,
            weight,
            bias,
            runtime.worklist_items,
            runtime.bundle_counts,
            runtime.max_bm,
            int(runtime.fixed_bundle_capacity),
            runtime.n_out,
            resolved_config_id,
            resolved_allow_tf32,
        )
    finally:
        if restore_tf32:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


def finalized_static_bundle_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    runtime: CompactTiledWorklist,
    *,
    config_id: int = 0,
    reverse: bool = False,
    n_out: int | None = None,
) -> torch.Tensor:
    if runtime.bundle_size <= 1:
        raise AssertionError("static bundle operator requires bundle_size > 1")
    resolved_n_out = int(runtime.n_out if n_out is None else n_out)
    if features.dtype == torch.float32 and weight.dtype == torch.float32:
        if config_id < 0:
            config_id = 0
        symbol = "gtsparse3d_finalize_static_bundle_conv3d_reverse_forward" if reverse else "gtsparse3d_finalize_static_bundle_conv3d_forward"
        return getattr(_C, symbol)(
            features,
            weight,
            bias,
            runtime.worklist_items,
            runtime.bundle_counts,
            runtime.bundle_size,
            runtime.max_bm,
            int(runtime.fixed_bundle_capacity),
            resolved_n_out,
            int(config_id),
        )
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("static bundle operator currently requires float16 features and weight, or float32 for the SIMT path")
    resolved_config_id = 0 if config_id < 0 else _resolve_finalize_fp16_local_config_id("bundle", int(config_id), expected_bundle_size=runtime.bundle_size)
    if config_id < 0:
        global_config_id = get_finalize_fp16_config_id(
            features.size(1),
            weight.size(0),
            int(runtime.offset_counts.numel()),
            features,
            weight,
            runtime,
            active_ratio=_active_ratio_from_rowmap(runtime.offset_counts, runtime.n_out),
            candidate_variants=("bundle",),
            candidate_bundle_sizes=(runtime.bundle_size,),
            bundle_runtime_map={int(runtime.bundle_size): runtime},
        )
        resolved_config_id = _resolve_finalize_fp16_local_config_id("bundle", global_config_id, expected_bundle_size=runtime.bundle_size)
    symbol = "gtsparse3d_finalize_static_bundle_conv3d_fp16_reverse_forward" if reverse else "gtsparse3d_finalize_static_bundle_conv3d_fp16_forward"
    return getattr(_C, symbol)(
        features,
        weight,
        bias,
        runtime.worklist_items,
        runtime.bundle_counts,
        runtime.bundle_size,
        runtime.max_bm,
        int(runtime.fixed_bundle_capacity),
        resolved_n_out,
        resolved_config_id,
    )


def build_subm_compact_tiled_worklist(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    bundle_size: int = 1,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> CompactTiledWorklist:
    row_inputs, row_masks, offset_counts, global_offset_support = _build_subm_rowmap_cached(
        st.indices,
        kernel_size,
        padding,
        dilation,
    )
    return build_compact_tiled_worklist_from_rowmap(
        row_inputs,
        row_masks,
        offset_counts,
        max_bm=max_bm,
        bundle_size=bundle_size,
        global_offset_support=global_offset_support,
        sort_by_local_mask=sort_by_local_mask,
        local_mask_sort_impl=local_mask_sort_impl,
    )


def build_subm_compact_tiled_worklist_direct_proto(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    bundle_size: int = 1,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> CompactTiledWorklist:
    if kernel_size != (3, 3, 3):
        raise ValueError("direct builder prototype currently requires kernel_size=(3, 3, 3)")
    if not hasattr(_C, "gtsparse3d_debug_build_subm_compact_worklist_direct"):
        raise RuntimeError("direct compact builder prototype is not available in this build")
    build_row_masks = bool(sort_by_local_mask and bundle_size > 1)
    coord_hashmap = (
        st.coord_hashmap
        if st.coord_hashmap is not None
        else torch.empty((0, 2), device=st.indices.device, dtype=torch.int64)
    )
    worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts = _C.gtsparse3d_debug_build_subm_compact_worklist_direct(
        st.indices,
        int(st.spatial_shape[0]), int(st.spatial_shape[1]), int(st.spatial_shape[2]),
        int(padding[0]), int(padding[1]), int(padding[2]),
        int(dilation[0]), int(dilation[1]), int(dilation[2]),
        int(bundle_size),
        int(max_bm),
        build_row_masks,
        coord_hashmap,
    )
    runtime = CompactTiledWorklist(
        worklist_items=worklist_items,
        bundle_counts=bundle_counts,
        num_items_valid=num_items_valid,
        row_inputs=torch.empty((0, 0), device=st.indices.device, dtype=torch.int32),
        row_masks=row_masks,
        offset_counts=offset_counts,
        global_offset_support=offset_counts,
        max_bm=int(max_bm),
        bundle_size=int(bundle_size),
        n_out=int(st.indices.size(0)),
        fixed_bundle_capacity=((int(st.indices.size(0)) + int(max_bm) - 1) // int(max_bm)) * int(max_bm),
        coord_hashmap=st.coord_hashmap,
        local_mask_sorted=False,
    )
    if sort_by_local_mask:
        _sort_fixed_capacity_runtime_by_local_mask_(
            runtime,
            local_mask_sort_impl=_validate_local_mask_sort_impl(local_mask_sort_impl),
        )
    return runtime


def build_subm_compact_tiled_worklist_soa_proto(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    bundle_size: int = 4,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> CompactTiledWorklist:
    if kernel_size != (3, 3, 3):
        raise ValueError("specialized SoA prototype currently requires kernel_size=(3, 3, 3)")
    if bundle_size not in (2, 4, 8):
        return build_subm_compact_tiled_worklist(
            st,
            kernel_size,
            padding,
            dilation,
            max_bm=max_bm,
            bundle_size=bundle_size,
            sort_by_local_mask=sort_by_local_mask,
            local_mask_sort_impl=local_mask_sort_impl,
        )
    if not hasattr(_C, "gtsparse3d_debug_build_subm_compact_worklist_soa"):
        raise RuntimeError("specialized SoA prototype is not available in this build")
    if hasattr(_C, "gtsparse3d_debug_build_subm_compact_worklist_soa_into"):
        workspace = _get_soa_workspace(device=st.indices.device, n_out=int(st.indices.size(0)), max_bm=int(max_bm), bundle_size=int(bundle_size))
        worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts = _C.gtsparse3d_debug_build_subm_compact_worklist_soa_into(
            st.indices,
            int(st.spatial_shape[0]), int(st.spatial_shape[1]), int(st.spatial_shape[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(dilation[0]), int(dilation[1]), int(dilation[2]),
            int(bundle_size),
            int(max_bm),
            workspace["worklist_items"],
            workspace["num_items_valid"],
            workspace["row_masks"],
            workspace["offset_counts"],
            workspace["bundle_counts"],
            workspace["bundle_starts"],
            workspace["bundle_write_ptrs"],
            workspace["bundle_inputs"],
            workspace["sort_keys_in"],
            workspace["sort_indices_in"],
            bool(sort_by_local_mask),
        )
    else:
        workspace = None
        worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts = _C.gtsparse3d_debug_build_subm_compact_worklist_soa(
            st.indices,
            int(st.spatial_shape[0]), int(st.spatial_shape[1]), int(st.spatial_shape[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(dilation[0]), int(dilation[1]), int(dilation[2]),
            int(bundle_size),
            int(max_bm),
        )
    empty_row_inputs = torch.empty((0, 0), device=st.indices.device, dtype=torch.int32)
    runtime = CompactTiledWorklist(
        worklist_items=worklist_items,
        bundle_counts=bundle_counts,
        num_items_valid=num_items_valid,
        row_inputs=empty_row_inputs,
        row_masks=row_masks,
        offset_counts=offset_counts,
        global_offset_support=offset_counts,
        max_bm=int(max_bm),
        bundle_size=int(bundle_size),
        n_out=int(st.indices.size(0)),
        fixed_bundle_capacity=((int(st.indices.size(0)) + int(max_bm) - 1) // int(max_bm)) * int(max_bm),
        local_mask_sorted=False,
    )
    if sort_by_local_mask:
        if workspace is None:
            raise RuntimeError("specialized SoA prototype local-mask sort requires caller-provided workspace support")
        _sort_bundle_segments_by_local_mask_(
            runtime,
            local_mask_sort_impl=_validate_local_mask_sort_impl(local_mask_sort_impl),
            bundle_starts=workspace["bundle_starts"],
            num_items_valid=workspace["num_items_valid"],
            sort_scratch_items=workspace["sort_scratch_items"],
            sort_keys_in=workspace["sort_keys_in"],
            sort_keys_out=workspace["sort_keys_out"],
            sort_indices_in=workspace["sort_indices_in"],
            sort_indices_out=workspace["sort_indices_out"],
            sort_segment_ends=workspace["sort_segment_ends"],
            fast_sort_chunk_starts=workspace["fast_sort_chunk_starts"],
            fast_sort_chunk_lengths=workspace["fast_sort_chunk_lengths"],
            fast_sort_segment_chunk_begin=workspace["fast_sort_segment_chunk_begin"],
            fast_sort_total_chunks_valid=workspace["fast_sort_total_chunks_valid"],
            fast_sort_chunk_bin_offsets=workspace["fast_sort_chunk_bin_offsets"],
        )
        workspace["worklist_items"], workspace["sort_scratch_items"] = (
            workspace["sort_scratch_items"],
            workspace["worklist_items"],
        )
        runtime.worklist_items = workspace["worklist_items"]
        runtime.local_mask_sorted = True
    return runtime


def build_subm_compact_tiled_worklist_bundle4_soa_proto(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> CompactTiledWorklist:
    return build_subm_compact_tiled_worklist_soa_proto(
        st,
        kernel_size,
        padding,
        dilation,
        max_bm=max_bm,
        bundle_size=4,
        sort_by_local_mask=sort_by_local_mask,
        local_mask_sort_impl=local_mask_sort_impl,
    )


def build_full_compact_tiled_worklist_soa_proto(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    bundle_size: int = 4,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> tuple[CompactTiledWorklist, torch.Tensor, list[int]]:
    if kernel_size != (3, 3, 3):
        raise ValueError("specialized full SoA prototype currently requires kernel_size=(3, 3, 3)")
    if bundle_size not in (2, 4, 8):
        return build_full_compact_tiled_worklist(
            st.indices,
            st.spatial_shape,
            kernel_size,
            stride,
            padding,
            dilation,
            max_bm=max_bm,
            bundle_size=bundle_size,
            sort_by_local_mask=sort_by_local_mask,
            local_mask_sort_impl=local_mask_sort_impl,
        )
    if not hasattr(_C, "gtsparse3d_debug_build_full_compact_worklist_soa"):
        raise RuntimeError("specialized full SoA prototype is not available in this build")

    out_spatial = [
        (int(st.spatial_shape[i]) + 2 * int(padding[i]) - int(dilation[i]) * (int(kernel_size[i]) - 1) - 1)
        // int(stride[i]) + 1
        for i in range(3)
    ]
    out_coords = _C.gtsparse3d_enumerate_output_coords(
        st.indices,
        out_spatial[0], out_spatial[1], out_spatial[2],
        int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2]),
        int(stride[0]), int(stride[1]), int(stride[2]),
        int(padding[0]), int(padding[1]), int(padding[2]),
        int(dilation[0]), int(dilation[1]), int(dilation[2]),
    )

    workspace = None
    if hasattr(_C, "gtsparse3d_debug_build_full_compact_worklist_soa_into"):
        workspace = _get_soa_workspace(device=st.indices.device, n_out=int(out_coords.size(0)), max_bm=int(max_bm), bundle_size=int(bundle_size))
        worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts = _C.gtsparse3d_debug_build_full_compact_worklist_soa_into(
            out_coords,
            st.indices,
            int(stride[0]), int(stride[1]), int(stride[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(dilation[0]), int(dilation[1]), int(dilation[2]),
            int(bundle_size),
            int(max_bm),
            workspace["worklist_items"],
            workspace["num_items_valid"],
            workspace["row_masks"],
            workspace["offset_counts"],
            workspace["bundle_counts"],
            workspace["bundle_starts"],
            workspace["bundle_write_ptrs"],
            workspace["bundle_inputs"],
            workspace["sort_keys_in"],
            workspace["sort_indices_in"],
            bool(sort_by_local_mask),
        )
    else:
        worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts = _C.gtsparse3d_debug_build_full_compact_worklist_soa(
            out_coords,
            st.indices,
            int(stride[0]), int(stride[1]), int(stride[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(dilation[0]), int(dilation[1]), int(dilation[2]),
            int(bundle_size),
            int(max_bm),
        )

    empty_row_inputs = torch.empty((0, 0), device=st.indices.device, dtype=torch.int32)
    runtime = CompactTiledWorklist(
        worklist_items=worklist_items,
        bundle_counts=bundle_counts,
        num_items_valid=num_items_valid,
        row_inputs=empty_row_inputs,
        row_masks=row_masks,
        offset_counts=offset_counts,
        global_offset_support=offset_counts,
        max_bm=int(max_bm),
        bundle_size=int(bundle_size),
        n_out=int(out_coords.size(0)),
        fixed_bundle_capacity=((int(out_coords.size(0)) + int(max_bm) - 1) // int(max_bm)) * int(max_bm),
        local_mask_sorted=False,
    )
    if sort_by_local_mask:
        if workspace is None:
            raise RuntimeError("full bundle4 SoA prototype local-mask sort requires caller-provided workspace support")
        _sort_bundle_segments_by_local_mask_(
            runtime,
            local_mask_sort_impl=_validate_local_mask_sort_impl(local_mask_sort_impl),
            bundle_starts=workspace["bundle_starts"],
            num_items_valid=workspace["num_items_valid"],
            sort_scratch_items=workspace["sort_scratch_items"],
            sort_keys_in=workspace["sort_keys_in"],
            sort_keys_out=workspace["sort_keys_out"],
            sort_indices_in=workspace["sort_indices_in"],
            sort_indices_out=workspace["sort_indices_out"],
            sort_segment_ends=workspace["sort_segment_ends"],
            fast_sort_chunk_starts=workspace["fast_sort_chunk_starts"],
            fast_sort_chunk_lengths=workspace["fast_sort_chunk_lengths"],
            fast_sort_segment_chunk_begin=workspace["fast_sort_segment_chunk_begin"],
            fast_sort_total_chunks_valid=workspace["fast_sort_total_chunks_valid"],
            fast_sort_chunk_bin_offsets=workspace["fast_sort_chunk_bin_offsets"],
        )
        workspace["worklist_items"], workspace["sort_scratch_items"] = (
            workspace["sort_scratch_items"],
            workspace["worklist_items"],
        )
        runtime.worklist_items = workspace["worklist_items"]
        runtime.local_mask_sorted = True
    return runtime, out_coords, out_spatial


def build_full_compact_tiled_worklist_bundle4_soa_proto(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> tuple[CompactTiledWorklist, torch.Tensor, list[int]]:
    return build_full_compact_tiled_worklist_soa_proto(
        st,
        kernel_size,
        stride,
        padding,
        dilation,
        max_bm=max_bm,
        bundle_size=4,
        sort_by_local_mask=sort_by_local_mask,
        local_mask_sort_impl=local_mask_sort_impl,
    )


def build_subm_compact_tiled_worklist_bundle4_dense_grid_proto(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    reorder_active_rows: bool = False,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> CompactTiledWorklist:
    if kernel_size != (3, 3, 3):
        raise ValueError("bundle4 dense-grid prototype currently requires kernel_size=(3, 3, 3)")
    if not hasattr(_C, "gtsparse3d_debug_build_subm_compact_worklist_bundle4_dense_grid"):
        raise RuntimeError("bundle4 dense-grid prototype is not available in this build")
    perm = _brick_row_reorder_perm(st.indices, spatial_shape=st.spatial_shape) if reorder_active_rows else None
    build_coords = st.indices if perm is None else st.indices.index_select(0, perm)
    workspace = None
    if hasattr(_C, "gtsparse3d_debug_build_subm_compact_worklist_bundle4_dense_grid_into"):
        workspace = _get_soa_workspace(device=st.indices.device, n_out=int(st.indices.size(0)), max_bm=int(max_bm), bundle_size=4)
        grid_workspace = _get_dense_index_grid_workspace(
            device=st.indices.device,
            batch_size=int(st.batch_size),
            spatial_shape=st.spatial_shape,
        )
        epoch = int(grid_workspace.get("epoch", 0)) + 1
        if epoch >= 256:
            grid_workspace["index_grid"].zero_()
            epoch = 1
        grid_workspace["epoch"] = epoch
        worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts = _C.gtsparse3d_debug_build_subm_compact_worklist_bundle4_dense_grid_into(
            build_coords,
            int(st.spatial_shape[0]), int(st.spatial_shape[1]), int(st.spatial_shape[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(dilation[0]), int(dilation[1]), int(dilation[2]),
            int(epoch),
            int(max_bm),
            grid_workspace["index_grid"],
            workspace["worklist_items"],
            workspace["num_items_valid"],
            workspace["row_masks"],
            workspace["offset_counts"],
            workspace["bundle_counts"],
            workspace["bundle_starts"],
            workspace["bundle_write_ptrs"],
            workspace["bundle_inputs"],
            workspace["sort_keys_in"],
            workspace["sort_indices_in"],
            bool(sort_by_local_mask),
        )
    else:
        worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts = _C.gtsparse3d_debug_build_subm_compact_worklist_bundle4_dense_grid(
            build_coords,
            int(st.spatial_shape[0]), int(st.spatial_shape[1]), int(st.spatial_shape[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(dilation[0]), int(dilation[1]), int(dilation[2]),
            1,
            int(max_bm),
        )
    empty_row_inputs = torch.empty((0, 0), device=st.indices.device, dtype=torch.int32)
    runtime = CompactTiledWorklist(
        worklist_items=worklist_items,
        bundle_counts=bundle_counts,
        num_items_valid=num_items_valid,
        row_inputs=empty_row_inputs,
        row_masks=row_masks,
        offset_counts=offset_counts,
        global_offset_support=offset_counts,
        max_bm=int(max_bm),
        bundle_size=4,
        n_out=int(st.indices.size(0)),
        fixed_bundle_capacity=((int(st.indices.size(0)) + int(max_bm) - 1) // int(max_bm)) * int(max_bm),
        local_mask_sorted=False,
    )
    if perm is not None:
        _remap_worklist_items_by_perm_(runtime, perm)
    if sort_by_local_mask:
        if workspace is None:
            raise RuntimeError("bundle4 dense-grid prototype local-mask sort requires caller-provided workspace support")
        _sort_bundle_segments_by_local_mask_(
            runtime,
            local_mask_sort_impl=_validate_local_mask_sort_impl(local_mask_sort_impl),
            bundle_starts=workspace["bundle_starts"],
            num_items_valid=workspace["num_items_valid"],
            sort_scratch_items=workspace["sort_scratch_items"],
            sort_keys_in=workspace["sort_keys_in"],
            sort_keys_out=workspace["sort_keys_out"],
            sort_indices_in=workspace["sort_indices_in"],
            sort_indices_out=workspace["sort_indices_out"],
            sort_segment_ends=workspace["sort_segment_ends"],
            fast_sort_chunk_starts=workspace["fast_sort_chunk_starts"],
            fast_sort_chunk_lengths=workspace["fast_sort_chunk_lengths"],
            fast_sort_segment_chunk_begin=workspace["fast_sort_segment_chunk_begin"],
            fast_sort_total_chunks_valid=workspace["fast_sort_total_chunks_valid"],
            fast_sort_chunk_bin_offsets=workspace["fast_sort_chunk_bin_offsets"],
        )
        workspace["worklist_items"], workspace["sort_scratch_items"] = (
            workspace["sort_scratch_items"],
            workspace["worklist_items"],
        )
        runtime.worklist_items = workspace["worklist_items"]
        runtime.local_mask_sorted = True
    return runtime


def build_full_compact_tiled_worklist(
    coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    bundle_size: int = 1,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
    batch_size: int = 1,
) -> tuple[CompactTiledWorklist, torch.Tensor, list[int]]:
    if not hasattr(_C, "gtsparse3d_debug_build_full_compact_worklist_direct"):
        raise RuntimeError("direct full compact builder prototype is not available in this build")
    k_vol = int(kernel_size[0] * kernel_size[1] * kernel_size[2])
    if k_vol <= 0 or k_vol > _MAX_FINALIZED_OFFSETS:
        raise ValueError(f"full direct builder currently requires kernel volume in [1, {_MAX_FINALIZED_OFFSETS}]")
    out_spatial = [
        (int(spatial_shape[i]) + 2 * int(padding[i]) - int(dilation[i]) * (int(kernel_size[i]) - 1) - 1)
        // int(stride[i]) + 1
        for i in range(3)
    ]
    build_row_masks = bool(sort_by_local_mask and bundle_size > 1)
    worklist_items, bundle_counts, num_items_valid, row_masks, offset_counts, out_coords, out_coord_hashmap = _C.gtsparse3d_debug_build_full_compact_worklist_direct(
        coords,
        out_spatial[0], out_spatial[1], out_spatial[2],
        int(kernel_size[0]), int(kernel_size[1]), int(kernel_size[2]),
        int(stride[0]), int(stride[1]), int(stride[2]),
        int(padding[0]), int(padding[1]), int(padding[2]),
        int(dilation[0]), int(dilation[1]), int(dilation[2]),
        int(bundle_size),
        int(max_bm),
        build_row_masks,
        int(batch_size),
    )
    runtime = CompactTiledWorklist(
        worklist_items=worklist_items,
        bundle_counts=bundle_counts,
        num_items_valid=num_items_valid,
        row_inputs=torch.empty((0, 0), device=coords.device, dtype=torch.int32),
        row_masks=row_masks,
        offset_counts=offset_counts,
        global_offset_support=offset_counts,
        max_bm=int(max_bm),
        bundle_size=int(bundle_size),
        n_out=int(out_coords.size(0)),
        fixed_bundle_capacity=((int(out_coords.size(0)) + int(max_bm) - 1) // int(max_bm)) * int(max_bm),
        coord_hashmap=out_coord_hashmap,
        local_mask_sorted=False,
    )
    if sort_by_local_mask:
        _sort_fixed_capacity_runtime_by_local_mask_(
            runtime,
            local_mask_sort_impl=_validate_local_mask_sort_impl(local_mask_sort_impl),
        )
    return runtime, out_coords, out_spatial


def build_inverse_compact_tiled_worklist(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    max_bm: int,
    bundle_size: int = 1,
    sort_by_local_mask: bool = False,
    local_mask_sort_impl: str = "auto",
) -> CompactTiledWorklist:
    row_inputs, row_masks, offset_counts, global_offset_support = _build_inverse_rowmap_cached(
        in_coords,
        out_coords,
        kernel_size,
        stride,
        padding,
        dilation,
    )
    return build_compact_tiled_worklist_from_rowmap(
        row_inputs,
        row_masks,
        offset_counts,
        max_bm=max_bm,
        bundle_size=bundle_size,
        global_offset_support=global_offset_support,
        sort_by_local_mask=sort_by_local_mask,
        local_mask_sort_impl=local_mask_sort_impl,
    )
