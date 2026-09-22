"""Expanded-worklist sparse convolution prototype."""

import math
from collections import defaultdict, deque
from itertools import combinations
from typing import Tuple

import torch

from gtsparse import _C
from gtsparse.sparse3d.ew_autotune import get_config_id

from .sparse_tensor import GTSparseSparseConvTensor


_SUBSET_MASK_BANK_CPU: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_SUBSET_MASK_BANK_DEVICE: dict[tuple[str, int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


# Baseline SIMT / explicit-schedule config table.
# Scheduled FP32 paths read their config table from CUDA via `_scheduled_config_table`.
EW_SIMT_CONFIGS = [
    {"BM": 64, "BN": 64, "BK": 8, "TM": 4, "TN": 4},
    {"BM": 64, "BN": 128, "BK": 8, "TM": 4, "TN": 4},
    {"BM": 128, "BN": 64, "BK": 8, "TM": 4, "TN": 4},
    {"BM": 128, "BN": 128, "BK": 8, "TM": 4, "TN": 4},
    {"BM": 64, "BN": 64, "BK": 16, "TM": 4, "TN": 4},
    {"BM": 64, "BN": 128, "BK": 16, "TM": 4, "TN": 4},
    {"BM": 128, "BN": 64, "BK": 16, "TM": 4, "TN": 4},
    {"BM": 128, "BN": 128, "BK": 16, "TM": 4, "TN": 4},
    {"BM": 64, "BN": 64, "BK": 32, "TM": 4, "TN": 4},
    {"BM": 64, "BN": 128, "BK": 32, "TM": 4, "TN": 4},
    {"BM": 128, "BN": 64, "BK": 32, "TM": 4, "TN": 4},
    {"BM": 128, "BN": 128, "BK": 32, "TM": 4, "TN": 4},
    {"BM": 128, "BN": 128, "BK": 8, "TM": 8, "TN": 8},
    {"BM": 128, "BN": 128, "BK": 16, "TM": 8, "TN": 8},
    {"BM": 128, "BN": 128, "BK": 32, "TM": 8, "TN": 8},
    {"BM": 64, "BN": 128, "BK": 16, "TM": 4, "TN": 8},
    {"BM": 128, "BN": 64, "BK": 16, "TM": 8, "TN": 4},
    {"BM": 64, "BN": 128, "BK": 32, "TM": 4, "TN": 8},
    {"BM": 128, "BN": 64, "BK": 32, "TM": 8, "TN": 4},
    {"BM": 32, "BN": 64, "BK": 8, "TM": 4, "TN": 4},
    {"BM": 32, "BN": 64, "BK": 16, "TM": 4, "TN": 4},
    {"BM": 32, "BN": 64, "BK": 32, "TM": 4, "TN": 4},
    {"BM": 32, "BN": 128, "BK": 8, "TM": 4, "TN": 4},
    {"BM": 32, "BN": 128, "BK": 16, "TM": 4, "TN": 4},
]

_SCHEDULED_CONFIG_TABLE_CACHE: dict[str, list[dict[str, int]]] = {}
_SCHEDULED_DEV_CONFIG_TABLE_CACHE: dict[str, list[dict[str, int]]] = {}
_SCHEDULED_BUNDLE_FP16_CONFIG_TABLE_CACHE: list[dict[str, int | bool]] | None = None
_SCHEDULED_BUNDLE_TILE_HEADER_CACHE: dict[tuple[int, int, int, int], torch.Tensor] = {}


def _scheduled_config_table(reuse_mode: str) -> list[dict[str, int]]:
    cached = _SCHEDULED_CONFIG_TABLE_CACHE.get(reuse_mode)
    if cached is not None:
        return cached
    if reuse_mode in {"off", "row_selective"}:
        table = [
            {str(k): int(v) for k, v in row.items()}
            for row in _C.gtsparse3d_expanded_worklist_conv3d_scheduled_config_table(2)
        ]
    else:
        raise ValueError("scheduled reuse_mode must be one of: off, row_selective")
    _SCHEDULED_CONFIG_TABLE_CACHE[reuse_mode] = table
    return table


def _scheduled_dev_config_table(reuse_mode: str) -> list[dict[str, int]]:
    cached = _SCHEDULED_DEV_CONFIG_TABLE_CACHE.get(reuse_mode)
    if cached is not None:
        return cached
    if reuse_mode in {"off", "row_selective"}:
        table = [
            {str(k): int(v) for k, v in row.items()}
            for row in _C.gtsparse3d_expanded_worklist_conv3d_scheduled_config_table_dev(2)
        ]
    else:
        raise ValueError("scheduled reuse_mode must be one of: off, row_selective")
    _SCHEDULED_DEV_CONFIG_TABLE_CACHE[reuse_mode] = table
    return table


def _scheduled_bundle_fp16_config_table() -> list[dict[str, int | bool]]:
    global _SCHEDULED_BUNDLE_FP16_CONFIG_TABLE_CACHE
    if _SCHEDULED_BUNDLE_FP16_CONFIG_TABLE_CACHE is not None:
        return _SCHEDULED_BUNDLE_FP16_CONFIG_TABLE_CACHE
    table = [
        {str(k): (bool(v) if k == "supported" else int(v)) for k, v in row.items()}
        for row in _C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_fp16_config_table()
    ]
    _SCHEDULED_BUNDLE_FP16_CONFIG_TABLE_CACHE = table
    return table
def scheduled_config_count(reuse_mode: str = "row_selective") -> int:
    return len(_scheduled_config_table(reuse_mode))


def _scheduled_exact_base_config_id(config_id: int) -> int:
    if config_id < 0:
        return 0
    return int(config_id % len(EW_SIMT_CONFIGS))


def _offset_starts_from_counts(offset_counts: torch.Tensor) -> torch.Tensor:
    starts = torch.empty_like(offset_counts)
    if offset_counts.numel() == 0:
        return starts
    starts[0] = 0
    if offset_counts.numel() > 1:
        starts[1:] = offset_counts.cumsum(0)[:-1]
    return starts


def compact_triplet_worklist(
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert padded offset-major pairs into compact explicit triplets."""
    K = offset_counts.numel()
    if K == 0:
        return pairs.new_empty((0, 3)), offset_counts

    N_stride = pairs.size(0) // K
    offset_starts = _offset_starts_from_counts(offset_counts)
    total = int(offset_counts.sum().item())
    triplets = pairs.new_empty((total, 3))

    cursor = 0
    for offset in range(K):
        count = int(offset_counts[offset].item())
        if count == 0:
            continue
        chunk = pairs[offset * N_stride: offset * N_stride + count]
        triplets[cursor: cursor + count, :2] = chunk
        triplets[cursor: cursor + count, 2] = offset
        cursor += count

    return triplets, offset_starts


def get_ew_simt_config(config_id: int) -> dict[str, int]:
    return EW_SIMT_CONFIGS[_scheduled_exact_base_config_id(config_id)]


def get_scheduled_simt_config(
    config_id: int,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    table = _scheduled_config_table(reuse_mode)
    if not table:
        raise RuntimeError(f"empty scheduled config table for reuse_mode={reuse_mode}")
    if config_id < 0 or config_id >= len(table) or not bool(table[config_id].get("supported", True)):
        config_id = 0
    row = table[config_id]
    return {
        "BM": int(row["BM"]),
        "BN": int(row["BN"]),
        "BK": int(row["BK"]),
        "TM": int(row["TM"]),
        "TN": int(row["TN"]),
    }


def get_scheduled_bundle_config(
    config_id: int,
    dtype: torch.dtype,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    if dtype == torch.float16:
        table = _scheduled_bundle_fp16_config_table()
    elif dtype == torch.float32:
        table = _scheduled_config_table(reuse_mode)
    else:
        raise TypeError(f"scheduled bundle path does not support dtype={dtype}")
    if not table:
        raise RuntimeError(f"empty scheduled bundle config table for dtype={dtype}")
    if config_id < 0 or config_id >= len(table):
        config_id = 0
    row = table[config_id]
    return {
        "BM": int(row["BM"]),
        "BN": int(row["BN"]),
        "BK": int(row["BK"]),
        "TM": int(row["TM"]),
        "TN": int(row["TN"]),
    }


def explicit_tile_schedule_launch_info(config_id: int) -> dict[str, int]:
    grid_dim_x, blocks_per_sm, num_sms, threads_per_block, smem_bytes = (
        _C.gtsparse3d_explicit_tile_schedule_launch_info(config_id)
    )
    return {
        "grid_dim_x": int(grid_dim_x),
        "blocks_per_sm": int(blocks_per_sm),
        "num_sms": int(num_sms),
        "threads_per_block": int(threads_per_block),
        "smem_bytes": int(smem_bytes),
    }


def scheduled_expanded_worklist_launch_info(
    config_id: int,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    reuse_mode_map = {
        "off": 2,
        "row_selective": 2,
    }
    if reuse_mode not in reuse_mode_map:
        raise ValueError("scheduled reuse_mode must be one of: off, row_selective")
    grid_dim_x, blocks_per_sm, num_sms, threads_per_block, smem_bytes = (
        _C.gtsparse3d_expanded_worklist_conv3d_scheduled_launch_info(
            config_id, reuse_mode_map[reuse_mode])
    )
    return {
        "grid_dim_x": int(grid_dim_x),
        "blocks_per_sm": int(blocks_per_sm),
        "num_sms": int(num_sms),
        "threads_per_block": int(threads_per_block),
        "smem_bytes": int(smem_bytes),
    }


def scheduled_expanded_worklist_dev_launch_info(
    config_id: int,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    reuse_mode_map = {
        "off": 2,
        "row_selective": 2,
    }
    if reuse_mode not in reuse_mode_map:
        raise ValueError("scheduled reuse_mode must be one of: off, row_selective")
    grid_dim_x, blocks_per_sm, num_sms, threads_per_block, smem_bytes = (
        _C.gtsparse3d_expanded_worklist_conv3d_scheduled_launch_info_dev(
            config_id, reuse_mode_map[reuse_mode])
    )
    return {
        "grid_dim_x": int(grid_dim_x),
        "blocks_per_sm": int(blocks_per_sm),
        "num_sms": int(num_sms),
        "threads_per_block": int(threads_per_block),
        "smem_bytes": int(smem_bytes),
    }


def scheduled_expanded_worklist_dev2_launch_info(
    config_id: int,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    reuse_mode_map = {
        "off": 2,
        "row_selective": 2,
    }
    if reuse_mode not in reuse_mode_map:
        raise ValueError("scheduled reuse_mode must be one of: off, row_selective")
    grid_dim_x, blocks_per_sm, num_sms, threads_per_block, smem_bytes = (
        _C.gtsparse3d_expanded_worklist_conv3d_scheduled_launch_info_dev2(
            config_id, reuse_mode_map[reuse_mode])
    )
    return {
        "grid_dim_x": int(grid_dim_x),
        "blocks_per_sm": int(blocks_per_sm),
        "num_sms": int(num_sms),
        "threads_per_block": int(threads_per_block),
        "smem_bytes": int(smem_bytes),
    }


def scheduled_bundle_launch_info(
    config_id: int,
    dtype: torch.dtype,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    if dtype == torch.float16:
        grid_dim_x, blocks_per_sm, num_sms, threads_per_block, smem_bytes = (
            _C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_fp16_launch_info(config_id)
        )
        return {
            "grid_dim_x": int(grid_dim_x),
            "blocks_per_sm": int(blocks_per_sm),
            "num_sms": int(num_sms),
            "threads_per_block": int(threads_per_block),
            "smem_bytes": int(smem_bytes),
        }
    if dtype == torch.float32:
        return scheduled_expanded_worklist_launch_info(config_id, reuse_mode=reuse_mode)
    raise TypeError(f"scheduled bundle launch info does not support dtype={dtype}")


def scheduled_expanded_worklist_kernel_attrs(
    config_id: int,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    reuse_mode_map = {
        "off": 2,
        "row_selective": 2,
    }
    if reuse_mode not in reuse_mode_map:
        raise ValueError("scheduled reuse_mode must be one of: off, row_selective")
    attrs = _C.gtsparse3d_expanded_worklist_conv3d_scheduled_kernel_attrs(
        config_id, reuse_mode_map[reuse_mode])
    return {str(k): int(v) for k, v in attrs.items()}


def scheduled_expanded_worklist_dev_kernel_attrs(
    config_id: int,
    reuse_mode: str = "row_selective",
) -> dict[str, int]:
    reuse_mode_map = {
        "off": 2,
        "row_selective": 2,
    }
    if reuse_mode not in reuse_mode_map:
        raise ValueError("scheduled reuse_mode must be one of: off, row_selective")
    attrs = _C.gtsparse3d_expanded_worklist_conv3d_scheduled_kernel_attrs_dev(
        config_id, reuse_mode_map[reuse_mode])
    return {str(k): int(v) for k, v in attrs.items()}


def _sort_positions_by_active_pattern(
    active_by_pos_and_offset: torch.Tensor,
    out_rows: list[int],
) -> list[int]:
    """Sort output positions by residual sparsity pattern, then row id."""
    num_positions, K_vol = active_by_pos_and_offset.shape
    popcount = active_by_pos_and_offset.sum(dim=1).tolist()

    if K_vol <= 63:
        bit_values = (1 << torch.arange(K_vol, dtype=torch.int64))
        bitmasks = (
            active_by_pos_and_offset.to(torch.int64) * bit_values.unsqueeze(0)
        ).sum(dim=1).tolist()
    else:
        bitmasks = [
            tuple(bool(v) for v in active_by_pos_and_offset[pos].tolist())
            for pos in range(num_positions)
        ]

    positions = [pos for pos in range(num_positions) if popcount[pos] > 0]
    positions.sort(key=lambda pos: (-popcount[pos], bitmasks[pos], out_rows[pos]))
    return positions


def _sort_positions_by_mask_key_tensor(
    masks: torch.Tensor,
    popcount: torch.Tensor,
    out_rows: torch.Tensor,
    K_vol: int,
) -> torch.Tensor:
    """Torch sort for positions by (popcount desc, mask asc, row asc)."""
    active_positions = torch.nonzero(popcount > 0, as_tuple=False).view(-1)
    if active_positions.numel() == 0:
        return active_positions
    mask_base = 1 << K_vol
    row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1
    keys = (
        ((K_vol + 1 - popcount[active_positions].to(torch.int64)) * mask_base)
        + masks[active_positions]
    ) * row_base + out_rows[active_positions]
    order = torch.argsort(keys)
    return active_positions[order]


def _get_subset_mask_bank(
    K_vol: int,
    max_subset_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached subset masks and their sizes on the requested device."""
    cpu_key = (K_vol, max_subset_size)
    if cpu_key not in _SUBSET_MASK_BANK_CPU:
        masks: list[int] = []
        sizes: list[int] = []
        for subset_size in range(max_subset_size, 1, -1):
            for subset in combinations(range(K_vol), subset_size):
                mask = 0
                for off in subset:
                    mask |= (1 << off)
                masks.append(mask)
                sizes.append(subset_size)
        _SUBSET_MASK_BANK_CPU[cpu_key] = (
            torch.tensor(masks, dtype=torch.int64),
            torch.tensor(sizes, dtype=torch.int16),
        )

    dev_index = device.index if device.index is not None else -1
    dev_key = (device.type, dev_index, K_vol, max_subset_size)
    if dev_key not in _SUBSET_MASK_BANK_DEVICE:
        masks_cpu, sizes_cpu = _SUBSET_MASK_BANK_CPU[cpu_key]
        _SUBSET_MASK_BANK_DEVICE[dev_key] = (
            masks_cpu.to(device=device),
            sizes_cpu.to(device=device),
        )
    return _SUBSET_MASK_BANK_DEVICE[dev_key]


def _mask_to_offsets(mask: int, K_vol: int) -> list[int]:
    return [off for off in range(K_vol) if (mask & (1 << off)) != 0]


def _pick_subset_first_candidate(
    residual_active: torch.Tensor,
    out_rows: torch.Tensor,
    BM: int,
    *,
    max_subset_size: int = 3,
) -> tuple[list[int], list[int]] | None:
    """Choose the strongest residual offset subset and return grouped positions.

    Returns `(subset_offsets, selected_positions)` where `selected_positions`
    has length `num_groups * BM` and is already ordered for chunking into BM-sized
    groups. Positions are chosen from the subset support set with a bias toward
    lower residual popcount, so more rigid positions are consumed first.
    """
    num_positions, K_vol = residual_active.shape
    if num_positions < BM or K_vol == 0:
        return None

    device = residual_active.device
    residual_popcount = residual_active.sum(dim=1, dtype=torch.int32)
    active_positions = torch.nonzero(residual_popcount > 0, as_tuple=False).view(-1)
    if int(active_positions.numel()) < BM:
        return None

    offset_support = residual_active.sum(dim=0).tolist()
    candidate_offsets = [off for off, count in enumerate(offset_support) if count >= BM]
    if not candidate_offsets:
        return None

    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    residual_masks = (residual_active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)

    best_subset: tuple[int, ...] | None = None
    best_subset_mask = 0
    best_score: tuple[int, int, int] | None = None

    max_k = min(max_subset_size, len(candidate_offsets))
    chunk_size = 512
    for subset_size in range(max_k, 1, -1):
        subset_list = list(combinations(candidate_offsets, subset_size))
        if not subset_list:
            continue
        subset_masks = torch.tensor(
            [sum(1 << off for off in subset) for subset in subset_list],
            dtype=torch.int64,
            device=device,
        )
        for start in range(0, len(subset_list), chunk_size):
            chunk_masks = subset_masks[start: start + chunk_size]
            if chunk_masks.numel() == 0:
                continue
            support_counts = (
                ((residual_masks.unsqueeze(1) & chunk_masks.unsqueeze(0)) == chunk_masks.unsqueeze(0))
                .sum(dim=0, dtype=torch.int32)
            )
            num_groups = support_counts // BM
            reuse_transitions = (subset_size - 1) * num_groups
            local_best_reuse = int(reuse_transitions.max().item())
            if local_best_reuse <= 0:
                continue

            local_best_indices = torch.nonzero(
                reuse_transitions == local_best_reuse,
                as_tuple=False,
            ).view(-1)
            if local_best_indices.numel() > 1:
                local_support = support_counts[local_best_indices]
                local_best_support = int(local_support.max().item())
                support_tie = torch.nonzero(
                    local_support == local_best_support,
                    as_tuple=False,
                ).view(-1)[0]
                best_idx_in_chunk = int(local_best_indices[support_tie].item())
            else:
                best_idx_in_chunk = int(local_best_indices[0].item())
                local_best_support = int(support_counts[best_idx_in_chunk].item())

            score = (local_best_reuse, subset_size, local_best_support)
            if best_score is None or score > best_score:
                best_score = score
                best_subset = subset_list[start + best_idx_in_chunk]
                best_subset_mask = int(chunk_masks[best_idx_in_chunk].item())

    if best_subset is None:
        return None

    support_positions = torch.nonzero(
        (residual_masks & best_subset_mask) == best_subset_mask,
        as_tuple=False,
    ).view(-1)
    if support_positions.numel() == 0:
        return None

    mask_base = 1 << K_vol
    row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1
    support_sort_keys = (
        (residual_popcount[support_positions].to(torch.int64) * mask_base + residual_masks[support_positions])
        * row_base
        + out_rows[support_positions]
    )
    support_order = torch.argsort(support_sort_keys)
    support_positions = support_positions[support_order]
    num_groups = int(support_positions.numel()) // BM
    selected_positions = support_positions[: num_groups * BM].tolist()
    return list(best_subset), selected_positions


def _pick_subset_first_candidate_gpu(
    residual_active: torch.Tensor,
    residual_masks: torch.Tensor,
    residual_popcount: torch.Tensor,
    out_rows: torch.Tensor,
    bit_values: torch.Tensor,
    subset_masks_bank: torch.Tensor,
    subset_sizes_bank: torch.Tensor,
    BM: int,
    *,
    chunk_size: int = 1024,
) -> tuple[int, torch.Tensor] | None:
    """GPU-oriented candidate picker that operates on packed residual masks."""
    active_count = int((residual_popcount > 0).sum().item())
    if active_count < BM:
        return None

    offset_support = residual_active.sum(dim=0, dtype=torch.int32)
    candidate_offset_mask = int(
        ((offset_support >= BM).to(torch.int64) * bit_values).sum().item()
    )
    if candidate_offset_mask == 0:
        return None

    valid_bank = (subset_masks_bank & candidate_offset_mask) == subset_masks_bank
    valid_masks = subset_masks_bank[valid_bank]
    valid_sizes = subset_sizes_bank[valid_bank].to(torch.int32)
    if valid_masks.numel() == 0:
        return None

    num_positions = int(residual_masks.numel())
    mask_base = 1 << bit_values.numel()
    score_base_support = num_positions + 1
    score_base_mask = mask_base

    best_subset_mask = 0
    best_score = None
    for start in range(0, int(valid_masks.numel()), chunk_size):
        chunk_masks = valid_masks[start: start + chunk_size]
        chunk_sizes = valid_sizes[start: start + chunk_size]
        support_counts = (
            ((residual_masks.unsqueeze(1) & chunk_masks.unsqueeze(0)) == chunk_masks.unsqueeze(0))
            .sum(dim=0, dtype=torch.int32)
        )
        num_groups = support_counts // BM
        reuse_transitions = (chunk_sizes - 1) * num_groups
        if int(reuse_transitions.max().item()) <= 0:
            continue
        composite = (
            (
                reuse_transitions.to(torch.int64) * 8
                + chunk_sizes.to(torch.int64)
            ) * score_base_support
            + support_counts.to(torch.int64)
        ) * score_base_mask - chunk_masks
        best_idx = int(torch.argmax(composite).item())
        candidate_score = int(composite[best_idx].item())
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_subset_mask = int(chunk_masks[best_idx].item())

    if best_subset_mask == 0:
        return None

    support_positions = torch.nonzero(
        (residual_masks & best_subset_mask) == best_subset_mask,
        as_tuple=False,
    ).view(-1)
    if support_positions.numel() < BM:
        return None

    row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1
    support_keys = (
        (residual_popcount[support_positions].to(torch.int64) * mask_base + residual_masks[support_positions])
        * row_base
        + out_rows[support_positions]
    )
    support_order = torch.argsort(support_keys)
    support_positions = support_positions[support_order]
    num_groups = int(support_positions.numel()) // BM
    return best_subset_mask, support_positions[: num_groups * BM]


def _flatten_schedule_columns_row_major(columns: list[list[int]]) -> list[int]:
    """Flatten per-block tile columns into the kernel's row-major schedule."""
    max_column_height = max((len(col) for col in columns), default=0)
    flat_tile_ids: list[int] = []
    for row in range(max_column_height):
        for col in columns:
            if row < len(col):
                flat_tile_ids.append(col[row])
    return flat_tile_ids


def _compute_tile_keep_cpu(
    scheduled_pairs: torch.Tensor,
    final_tile_is_tail: torch.Tensor,
    BM: int,
    num_blocks: int,
) -> torch.Tensor:
    """Precompute exact-keep transitions for the full-row stage; tail always flushes."""
    num_tiles = int(scheduled_pairs.size(0) // BM)
    tile_keep = torch.zeros((num_tiles,), dtype=torch.uint8)
    if num_tiles == 0 or num_blocks <= 0:
        return tile_keep

    row_tiles = scheduled_pairs.view(num_tiles, BM, 2)
    full_row_iters = num_tiles // num_blocks
    if full_row_iters <= 1:
        return tile_keep

    for block in range(num_blocks):
        for row_iter in range(full_row_iters - 1):
            cur_tile = block + row_iter * num_blocks
            next_tile = cur_tile + num_blocks
            if bool(final_tile_is_tail[cur_tile].item()) or bool(final_tile_is_tail[next_tile].item()):
                continue
            prev = row_tiles[cur_tile]
            curr = row_tiles[next_tile]
            valid = (prev[:, 0] >= 0) & (curr[:, 0] >= 0)
            valid_count = int(valid.sum().item())
            if valid_count <= 0:
                continue
            if bool(torch.all(prev[valid, 1] == curr[valid, 1]).item()):
                tile_keep[cur_tile] = 1
    return tile_keep


def _compute_tile_keep_gpu(
    scheduled_pairs: torch.Tensor,
    final_tile_is_tail: torch.Tensor,
    BM: int,
    num_blocks: int,
) -> torch.Tensor:
    """GPU version of exact-keep precompute for the full-row stage; tail always flushes."""
    device = scheduled_pairs.device
    num_tiles = int(scheduled_pairs.size(0) // BM)
    tile_keep = torch.zeros((num_tiles,), dtype=torch.uint8, device=device)
    if num_tiles == 0 or num_blocks <= 0:
        return tile_keep

    full_row_iters = num_tiles // num_blocks
    if full_row_iters <= 1:
        return tile_keep

    row_tiles = scheduled_pairs.view(num_tiles, BM, 2)
    tile_ids = torch.arange(
        full_row_iters * num_blocks, device=device, dtype=torch.long).view(full_row_iters, num_blocks)
    cur_ids = tile_ids[:-1].reshape(-1)
    next_ids = tile_ids[1:].reshape(-1)
    non_tail = (~final_tile_is_tail[cur_ids]) & (~final_tile_is_tail[next_ids])
    if not bool(non_tail.any().item()):
        return tile_keep

    cur_ids = cur_ids[non_tail]
    next_ids = next_ids[non_tail]
    prev = row_tiles[cur_ids]
    curr = row_tiles[next_ids]
    valid = (prev[:, :, 0] >= 0) & (curr[:, :, 0] >= 0)
    valid_counts = valid.sum(dim=1)
    exact = ((prev[:, :, 1] == curr[:, :, 1]) | (~valid)).all(dim=1) & (valid_counts > 0)
    if exact.numel() > 0:
        tile_keep[cur_ids[exact]] = 1
    return tile_keep


def _build_schedule_from_tile_batches_cpu(
    tile_pairs_list: list[list[list[int]]],
    tile_offsets_list: list[int],
    tile_group_ids_list: list[int],
    columns: list[list[int]],
    stage1_num_tiles: int,
    total_pairs: int,
    num_positions: int,
    BM: int,
    num_blocks: int,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    num_rounds: int,
) -> dict[str, torch.Tensor | dict[str, float]]:
    stage1_columns = [
        [tile_id for tile_id in col if tile_id < stage1_num_tiles]
        for col in columns
    ]
    stage1_flat_tile_ids = _flatten_schedule_columns_row_major(stage1_columns)

    if not stage1_flat_tile_ids and total_pairs == 0:
        empty_pairs = torch.empty((0, 2), dtype=pair_dtype)
        empty_offsets = torch.empty((0,), dtype=offset_dtype)
        empty_groups = torch.empty((0,), dtype=torch.int32)
        empty_tail = torch.empty((0,), dtype=torch.bool)
        stats = {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": float(num_positions),
            "hole_count": 0.0,
            "stage1_hole_count": 0.0,
            "stage1_fill_ratio": 0.0,
            "stage1_num_tiles": 0.0,
            "tail_num_tiles": 0.0,
            "num_rounds": float(num_rounds),
        }
        return {
            "scheduled_pairs_cpu": empty_pairs,
            "tile_offsets_cpu": empty_offsets,
            "tile_keep_cpu": torch.empty((0,), dtype=torch.uint8),
            "stage1_pairs_cpu": empty_pairs.clone(),
            "tile_group_ids_cpu": empty_groups,
            "final_tile_is_tail_cpu": empty_tail,
            "stats": stats,
        }

    if stage1_flat_tile_ids:
        stage1_ordered_pairs = [
            [pair[:] for pair in tile_pairs_list[tile_id]]
            for tile_id in stage1_flat_tile_ids
        ]
        stage1_pairs_cpu = torch.tensor(stage1_ordered_pairs, dtype=pair_dtype).view(-1, 2)
    else:
        stage1_pairs_cpu = torch.empty((0, 2), dtype=pair_dtype)
    stage1_valid_pairs = int((stage1_pairs_cpu[:, 0] >= 0).sum().item())
    stage1_hole_count = int(stage1_pairs_cpu.size(0) - stage1_valid_pairs)

    final_flat_tile_ids = _flatten_schedule_columns_row_major(columns)
    ordered_pairs = [tile_pairs_list[tile_id] for tile_id in final_flat_tile_ids]
    ordered_offsets = [tile_offsets_list[tile_id] for tile_id in final_flat_tile_ids]
    ordered_group_ids = [tile_group_ids_list[tile_id] for tile_id in final_flat_tile_ids]
    ordered_tail_flags = [tile_id >= stage1_num_tiles for tile_id in final_flat_tile_ids]
    scheduled_pairs_cpu = torch.tensor(ordered_pairs, dtype=pair_dtype).view(-1, 2)
    tile_offsets_cpu = torch.tensor(ordered_offsets, dtype=offset_dtype)
    tile_group_ids_cpu = torch.tensor(ordered_group_ids, dtype=torch.int32)
    final_tile_is_tail_cpu = torch.tensor(ordered_tail_flags, dtype=torch.bool)
    tile_keep_cpu = _compute_tile_keep_cpu(
        scheduled_pairs_cpu,
        final_tile_is_tail_cpu,
        BM,
        num_blocks,
    )

    valid_pairs = int((scheduled_pairs_cpu[:, 0] >= 0).sum().item())
    hole_count = int(scheduled_pairs_cpu.size(0) - valid_pairs)
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"minimal scheduled worklist lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    stats = {
        "num_tiles": float(tile_offsets_cpu.numel()),
        "num_blocks": float(num_blocks),
        "padding_ratio": float(scheduled_pairs_cpu.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(scheduled_pairs_cpu.size(0), 1)),
        "num_positions": float(num_positions),
        "hole_count": float(hole_count),
        "stage1_hole_count": float(stage1_hole_count),
        "stage1_fill_ratio": float(stage1_valid_pairs / max(stage1_pairs_cpu.size(0), 1)),
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(len(tile_pairs_list) - stage1_num_tiles),
        "num_rounds": float(num_rounds),
    }
    return {
        "scheduled_pairs_cpu": scheduled_pairs_cpu,
        "tile_offsets_cpu": tile_offsets_cpu,
        "tile_keep_cpu": tile_keep_cpu,
        "stage1_pairs_cpu": stage1_pairs_cpu,
        "tile_group_ids_cpu": tile_group_ids_cpu,
        "final_tile_is_tail_cpu": final_tile_is_tail_cpu,
        "stats": stats,
    }


def _emit_exact_group_tiles_cpu(
    group_positions: list[int],
    group_mask: int,
    inputs_by_pos_and_offset: torch.Tensor,
    out_rows_cpu: torch.Tensor,
    tile_pairs_list: list[list[list[int]]],
    tile_offsets_list: list[int],
    tile_group_ids_list: list[int],
    columns: list[list[int]],
    column_lengths: list[int],
    next_group_id: int,
) -> int:
    subset_offsets = _mask_to_offsets(group_mask, int(inputs_by_pos_and_offset.size(1)))
    if len(subset_offsets) <= 1:
        return next_group_id
    group_tile_ids: list[int] = []
    group_id = next_group_id
    next_group_id += 1
    for off in subset_offsets:
        tile_id = len(tile_pairs_list)
        tile_offsets_list.append(int(off))
        tile_group_ids_list.append(group_id)
        tile_pairs = []
        for pos in group_positions:
            input_row = int(inputs_by_pos_and_offset[pos, off].item())
            if input_row < 0:
                raise RuntimeError(
                    "exact group stage produced an invalid slot; schedule invariant broken")
            tile_pairs.append([input_row, int(out_rows_cpu[pos].item())])
        tile_pairs_list.append(tile_pairs)
        group_tile_ids.append(tile_id)
    if group_tile_ids:
        target_col = min(range(len(columns)), key=lambda col: column_lengths[col])
        columns[target_col].extend(group_tile_ids)
        column_lengths[target_col] += len(group_tile_ids)
    return next_group_id


def _support_sorted_mask_order_cpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows_cpu: torch.Tensor,
) -> list[int]:
    K_vol = int(residual_active.size(1))
    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1)
    if active_positions.numel() == 0:
        return []

    reordered_masks = _support_reordered_masks_cpu(
        residual_masks,
        residual_active,
    )

    K_base = 1 << K_vol
    row_base = int(out_rows_cpu.max().item()) + 1 if out_rows_cpu.numel() > 0 else 1
    popcount = residual_active.sum(dim=1, dtype=torch.int64)
    keys = (
        ((K_vol + 1 - popcount[active_positions]) * K_base)
        + reordered_masks[active_positions]
    ) * row_base + out_rows_cpu[active_positions]
    order = torch.argsort(keys)
    return active_positions[order].tolist()


def _support_sorted_mask_order_tensor(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows: torch.Tensor,
) -> torch.Tensor:
    K_vol = int(residual_active.size(1))
    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1)
    if active_positions.numel() == 0:
        return active_positions

    reordered_masks = _support_reordered_masks_tensor(
        residual_masks,
        residual_active,
    )

    K_base = 1 << K_vol
    row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1
    popcount = residual_active.sum(dim=1, dtype=torch.int64)
    keys = (
        ((K_vol + 1 - popcount[active_positions]) * K_base)
        + reordered_masks[active_positions]
    ) * row_base + out_rows[active_positions]
    order = torch.argsort(keys)
    return active_positions[order]


def _support_reordered_masks_cpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
) -> torch.Tensor:
    offset_support = residual_active.sum(dim=0, dtype=torch.int64)
    support_rank = torch.argsort(offset_support, stable=True)
    reordered_masks = torch.zeros_like(residual_masks, dtype=torch.int64)
    for new_bit, off in enumerate(support_rank.tolist()):
        bit = 1 << off
        reordered_masks |= (((residual_masks & bit) != 0).to(torch.int64) << new_bit)
    return reordered_masks


def _support_reordered_masks_tensor(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
) -> torch.Tensor:
    device = residual_masks.device
    K_vol = int(residual_active.size(1))
    offset_support = residual_active.sum(dim=0, dtype=torch.int64)
    support_rank = torch.argsort(offset_support, stable=True)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    reordered_bits = (
        (residual_masks.unsqueeze(1) & (1 << support_rank).unsqueeze(0)) != 0
    ).to(torch.int64)
    return (reordered_bits * bit_values.unsqueeze(0)).sum(dim=1)


def _dplane_popcount_order_cpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows_cpu: torch.Tensor,
) -> list[int]:
    """Sort by total popcount, then 3 depth-plane popcounts, then raw mask."""
    K_vol = int(residual_active.size(1))
    if K_vol != 27:
        popcount = residual_active.sum(dim=1, dtype=torch.int64)
        return _sort_positions_by_mask_key_tensor(
            residual_masks,
            popcount,
            out_rows_cpu,
            K_vol,
        ).tolist()

    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1).tolist()
    if not active_positions:
        return []

    popcount = residual_active.sum(dim=1).tolist()
    plane0 = residual_active[:, 0:9].sum(dim=1).tolist()
    plane1 = residual_active[:, 9:18].sum(dim=1).tolist()
    plane2 = residual_active[:, 18:27].sum(dim=1).tolist()
    masks = residual_masks.tolist()
    out_rows = out_rows_cpu.tolist()

    active_positions.sort(
        key=lambda pos: (
            -popcount[pos],
            -plane0[pos],
            -plane1[pos],
            -plane2[pos],
            masks[pos],
            out_rows[pos],
        )
    )
    return active_positions


def _dplane_popcount_order_tensor(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows: torch.Tensor,
) -> torch.Tensor:
    """GPU sort for the 3x3x3 depth-plane-popcount key."""
    K_vol = int(residual_active.size(1))
    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1)
    if active_positions.numel() == 0:
        return active_positions
    if K_vol != 27:
        popcount = residual_active.sum(dim=1, dtype=torch.int64)
        return _sort_positions_by_mask_key_tensor(
            residual_masks,
            popcount,
            out_rows,
            K_vol,
        )

    popcount = residual_active.sum(dim=1, dtype=torch.int64)
    plane0 = residual_active[:, 0:9].sum(dim=1, dtype=torch.int64)
    plane1 = residual_active[:, 9:18].sum(dim=1, dtype=torch.int64)
    plane2 = residual_active[:, 18:27].sum(dim=1, dtype=torch.int64)

    order = active_positions
    order = order[torch.argsort(out_rows[order], stable=True)]
    order = order[torch.argsort(residual_masks[order], stable=True)]
    order = order[torch.argsort(plane2[order], descending=True, stable=True)]
    order = order[torch.argsort(plane1[order], descending=True, stable=True)]
    order = order[torch.argsort(plane0[order], descending=True, stable=True)]
    order = order[torch.argsort(popcount[order], descending=True, stable=True)]
    return order


def _dplane_support_sorted_order_cpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows_cpu: torch.Tensor,
) -> list[int]:
    """Sort by 3 depth-plane densities, then support-ranked mask identity."""
    K_vol = int(residual_active.size(1))
    if K_vol != 27:
        return _support_sorted_mask_order_cpu(
            residual_masks,
            residual_active,
            out_rows_cpu,
        )

    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1).tolist()
    if not active_positions:
        return []

    plane0 = residual_active[:, 0:9].sum(dim=1).tolist()
    plane1 = residual_active[:, 9:18].sum(dim=1).tolist()
    plane2 = residual_active[:, 18:27].sum(dim=1).tolist()
    reordered_masks = _support_reordered_masks_cpu(
        residual_masks,
        residual_active,
    ).tolist()
    out_rows = out_rows_cpu.tolist()

    active_positions.sort(
        key=lambda pos: (
            -plane0[pos],
            -plane1[pos],
            -plane2[pos],
            reordered_masks[pos],
            out_rows[pos],
        )
    )
    return active_positions


def _dplane_support_sorted_order_tensor(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows: torch.Tensor,
) -> torch.Tensor:
    """GPU sort by 3 depth-plane densities, then support-ranked mask identity."""
    K_vol = int(residual_active.size(1))
    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1)
    if active_positions.numel() == 0:
        return active_positions
    if K_vol != 27:
        return _support_sorted_mask_order_tensor(
            residual_masks,
            residual_active,
            out_rows,
        )

    plane0 = residual_active[:, 0:9].sum(dim=1, dtype=torch.int64)
    plane1 = residual_active[:, 9:18].sum(dim=1, dtype=torch.int64)
    plane2 = residual_active[:, 18:27].sum(dim=1, dtype=torch.int64)
    reordered_masks = _support_reordered_masks_tensor(
        residual_masks,
        residual_active,
    )

    order = active_positions
    order = order[torch.argsort(out_rows[order], stable=True)]
    order = order[torch.argsort(reordered_masks[order], stable=True)]
    order = order[torch.argsort(plane2[order], descending=True, stable=True)]
    order = order[torch.argsort(plane1[order], descending=True, stable=True)]
    order = order[torch.argsort(plane0[order], descending=True, stable=True)]
    return order


def _axis_marginal_support_sorted_order_cpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows_cpu: torch.Tensor,
) -> list[int]:
    """Sort by d/h/w marginal densities, then support-ranked mask identity."""
    K_vol = int(residual_active.size(1))
    if K_vol != 27:
        return _support_sorted_mask_order_cpu(
            residual_masks,
            residual_active,
            out_rows_cpu,
        )

    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1).tolist()
    if not active_positions:
        return []

    marginals = []
    # depth slices
    marginals.extend([
        residual_active[:, 0:9].sum(dim=1).tolist(),
        residual_active[:, 9:18].sum(dim=1).tolist(),
        residual_active[:, 18:27].sum(dim=1).tolist(),
    ])
    # height slices
    h_slices = (
        [0, 1, 2, 9, 10, 11, 18, 19, 20],
        [3, 4, 5, 12, 13, 14, 21, 22, 23],
        [6, 7, 8, 15, 16, 17, 24, 25, 26],
    )
    for idxs in h_slices:
        marginals.append(residual_active[:, idxs].sum(dim=1).tolist())
    # width slices
    w_slices = (
        [0, 3, 6, 9, 12, 15, 18, 21, 24],
        [1, 4, 7, 10, 13, 16, 19, 22, 25],
        [2, 5, 8, 11, 14, 17, 20, 23, 26],
    )
    for idxs in w_slices:
        marginals.append(residual_active[:, idxs].sum(dim=1).tolist())

    reordered_masks = _support_reordered_masks_cpu(
        residual_masks,
        residual_active,
    ).tolist()
    out_rows = out_rows_cpu.tolist()

    active_positions.sort(
        key=lambda pos: (
            -marginals[0][pos],
            -marginals[1][pos],
            -marginals[2][pos],
            -marginals[3][pos],
            -marginals[4][pos],
            -marginals[5][pos],
            -marginals[6][pos],
            -marginals[7][pos],
            -marginals[8][pos],
            reordered_masks[pos],
            out_rows[pos],
        )
    )
    return active_positions


def _axis_marginal_support_sorted_order_tensor(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows: torch.Tensor,
) -> torch.Tensor:
    """GPU sort by d/h/w marginal densities, then support-ranked mask identity."""
    K_vol = int(residual_active.size(1))
    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1)
    if active_positions.numel() == 0:
        return active_positions
    if K_vol != 27:
        return _support_sorted_mask_order_tensor(
            residual_masks,
            residual_active,
            out_rows,
        )

    marginals = [
        residual_active[:, 0:9].sum(dim=1, dtype=torch.int64),
        residual_active[:, 9:18].sum(dim=1, dtype=torch.int64),
        residual_active[:, 18:27].sum(dim=1, dtype=torch.int64),
        residual_active[:, [0, 1, 2, 9, 10, 11, 18, 19, 20]].sum(dim=1, dtype=torch.int64),
        residual_active[:, [3, 4, 5, 12, 13, 14, 21, 22, 23]].sum(dim=1, dtype=torch.int64),
        residual_active[:, [6, 7, 8, 15, 16, 17, 24, 25, 26]].sum(dim=1, dtype=torch.int64),
        residual_active[:, [0, 3, 6, 9, 12, 15, 18, 21, 24]].sum(dim=1, dtype=torch.int64),
        residual_active[:, [1, 4, 7, 10, 13, 16, 19, 22, 25]].sum(dim=1, dtype=torch.int64),
        residual_active[:, [2, 5, 8, 11, 14, 17, 20, 23, 26]].sum(dim=1, dtype=torch.int64),
    ]
    reordered_masks = _support_reordered_masks_tensor(
        residual_masks,
        residual_active,
    )

    order = active_positions
    order = order[torch.argsort(out_rows[order], stable=True)]
    order = order[torch.argsort(reordered_masks[order], stable=True)]
    for marginal in reversed(marginals):
        order = order[torch.argsort(marginal[order], descending=True, stable=True)]
    return order


def _single_round_exact_group_order_cpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows_cpu: torch.Tensor,
    *,
    order_mode: str,
) -> list[int]:
    if order_mode == "support_sorted_mask":
        return _support_sorted_mask_order_cpu(
            residual_masks,
            residual_active,
            out_rows_cpu,
        )
    if order_mode == "dplane_popcount":
        return _dplane_popcount_order_cpu(
            residual_masks,
            residual_active,
            out_rows_cpu,
        )
    if order_mode == "dplane_support_sorted":
        return _dplane_support_sorted_order_cpu(
            residual_masks,
            residual_active,
            out_rows_cpu,
        )
    if order_mode == "axis_marginal_support_sorted":
        return _axis_marginal_support_sorted_order_cpu(
            residual_masks,
            residual_active,
            out_rows_cpu,
        )
    raise ValueError(f"unknown exact-group order_mode: {order_mode}")


def _single_round_exact_group_order_tensor(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows: torch.Tensor,
    *,
    order_mode: str,
) -> torch.Tensor:
    if order_mode == "support_sorted_mask":
        return _support_sorted_mask_order_tensor(
            residual_masks,
            residual_active,
            out_rows,
        )
    if order_mode == "dplane_popcount":
        return _dplane_popcount_order_tensor(
            residual_masks,
            residual_active,
            out_rows,
        )
    if order_mode == "dplane_support_sorted":
        return _dplane_support_sorted_order_tensor(
            residual_masks,
            residual_active,
            out_rows,
        )
    if order_mode == "axis_marginal_support_sorted":
        return _axis_marginal_support_sorted_order_tensor(
            residual_masks,
            residual_active,
            out_rows,
        )
    raise ValueError(f"unknown exact-group order_mode: {order_mode}")


def _position_fill_order_cpu(
    masks: torch.Tensor,
    active: torch.Tensor,
    out_rows: torch.Tensor,
    *,
    order_mode: str,
) -> list[int]:
    if order_mode == "popcount_mask":
        return _sort_positions_by_active_pattern(active, out_rows.tolist())
    if order_mode == "support_sorted_mask":
        return _support_sorted_mask_order_cpu(masks, active, out_rows)
    raise ValueError(f"unknown position-fill order_mode: {order_mode}")


def _position_fill_order_tensor(
    masks: torch.Tensor,
    active: torch.Tensor,
    out_rows: torch.Tensor,
    *,
    order_mode: str,
) -> torch.Tensor:
    K_vol = int(active.size(1))
    popcount = active.sum(dim=1, dtype=torch.int64)
    if order_mode == "popcount_mask":
        return _sort_positions_by_mask_key_tensor(masks, popcount, out_rows, K_vol)
    if order_mode == "support_sorted_mask":
        return _support_sorted_mask_order_tensor(masks, active, out_rows)
    raise ValueError(f"unknown position-fill order_mode: {order_mode}")


def _ordered_offsets_by_group_support_cpu(
    group_active: torch.Tensor,
    global_offset_support: torch.Tensor,
) -> list[int]:
    group_support = group_active.sum(dim=0, dtype=torch.int64)
    positive_offsets = torch.nonzero(group_support > 0, as_tuple=False).view(-1)
    if positive_offsets.numel() == 0:
        return []
    order = positive_offsets
    order = order[torch.argsort(order, stable=True)]
    order = order[torch.argsort(global_offset_support[order], descending=True, stable=True)]
    order = order[torch.argsort(group_support[order], descending=True, stable=True)]
    return order.tolist()


def _ordered_offsets_by_group_support_tensor(
    group_active: torch.Tensor,
    global_offset_support: torch.Tensor,
) -> torch.Tensor:
    group_support = group_active.sum(dim=0, dtype=torch.int64)
    positive_offsets = torch.nonzero(group_support > 0, as_tuple=False).view(-1)
    if positive_offsets.numel() == 0:
        return positive_offsets
    order = positive_offsets
    order = order[torch.argsort(order, stable=True)]
    order = order[torch.argsort(global_offset_support[order], descending=True, stable=True)]
    order = order[torch.argsort(group_support[order], descending=True, stable=True)]
    return order


def _build_position_fill_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    order_mode: str,
    min_keep_support: int,
) -> dict[str, torch.Tensor | dict[str, float]]:
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    active = inputs_by_pos_and_offset.ge(0)
    residual_active = active.clone()
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64))
    masks = (active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    position_order = _position_fill_order_cpu(
        masks,
        active,
        unique_out_rows.to(dtype=torch.int64),
        order_mode=order_mode,
    )
    global_offset_support = active.sum(dim=0, dtype=torch.int64)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pairs_list: list[list[list[int]]] = []
    tile_offsets_list: list[int] = []
    tile_group_ids_list: list[int] = []
    next_group_id = 0

    num_full_groups = len(position_order) // BM
    for group_idx in range(num_full_groups):
        positions = position_order[group_idx * BM: (group_idx + 1) * BM]
        group_active = residual_active[positions]
        ordered_offsets = _ordered_offsets_by_group_support_cpu(
            group_active,
            global_offset_support,
        )
        keep_offsets = [
            off for off in ordered_offsets
            if int(group_active[:, off].sum().item()) >= min_keep_support
        ]
        if not keep_offsets:
            continue

        tile_base = len(tile_pairs_list)
        for off in keep_offsets:
            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pos in enumerate(positions):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row >= 0:
                    tile_pairs[slot] = [input_row, int(unique_out_rows[pos].item())]
                    residual_active[pos, off] = False
            tile_pairs_list.append(tile_pairs)
            tile_offsets_list.append(int(off))
            tile_group_ids_list.append(next_group_id)

        target_col = min(range(num_blocks), key=column_lengths.__getitem__)
        group_tile_ids = list(range(tile_base, len(tile_pairs_list)))
        columns[target_col].extend(group_tile_ids)
        column_lengths[target_col] += len(group_tile_ids)
        next_group_id += 1

    stage1_num_tiles = len(tile_pairs_list)

    for off in range(K_vol):
        leftover_positions = [
            pos for pos in position_order
            if bool(residual_active[pos, off].item())
        ]
        for start in range(0, len(leftover_positions), BM):
            chunk = leftover_positions[start: start + BM]
            if not chunk:
                continue
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1

            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pos in enumerate(chunk):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "position-fill tail encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(unique_out_rows[pos].item())]
                residual_active[pos, off] = False
            tile_pairs_list.append(tile_pairs)

            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1

    if residual_active.any():
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"position-fill schedule left residual pairs after tail fill: {leftover_pairs}")

    debug = _build_schedule_from_tile_batches_cpu(
        tile_pairs_list,
        tile_offsets_list,
        tile_group_ids_list,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        pair_dtype,
        offset_dtype,
        1,
    )
    return debug


def _move_schedule_debug_to_device(
    debug: dict[str, torch.Tensor | dict[str, float]],
    device: torch.device,
) -> dict[str, torch.Tensor | dict[str, float]]:
    moved: dict[str, torch.Tensor | dict[str, float]] = {}
    for key, value in debug.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _build_slot_matrix_direct_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    active = inputs_by_pos_and_offset.ge(0)
    position_order = _sort_positions_by_active_pattern(active, unique_out_rows.tolist())

    slot_streams: list[list[list[int] | None]] = [[] for _ in range(BM)]
    for idx, pos in enumerate(position_order):
        slot_id = idx % BM
        active_offsets = torch.nonzero(active[pos], as_tuple=False).view(-1).tolist()
        if not active_offsets:
            continue
        current_phase = len(slot_streams[slot_id]) % K_vol
        max_delta = max(((off - current_phase) % K_vol) for off in active_offsets)
        out_row = int(unique_out_rows[pos].item())
        for delta in range(max_delta + 1):
            off = (current_phase + delta) % K_vol
            input_row = int(inputs_by_pos_and_offset[pos, off].item())
            if input_row >= 0:
                slot_streams[slot_id].append([input_row, out_row])
            else:
                slot_streams[slot_id].append(None)

    matrix_height = max((len(stream) for stream in slot_streams), default=0)
    if matrix_height == 0:
        empty_pairs = triplets_cpu.new_empty((0, 2))
        empty_offsets = offset_counts_cpu.new_empty((0,))
        empty_groups = torch.empty((0,), dtype=torch.int32)
        empty_tail = torch.empty((0,), dtype=torch.bool)
        stats = {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": float(num_positions),
            "hole_count": 0.0,
            "stage1_hole_count": 0.0,
            "stage1_fill_ratio": 0.0,
            "stage1_num_tiles": 0.0,
            "tail_num_tiles": 0.0,
            "num_rounds": 1.0,
        }
        return {
            "scheduled_pairs_cpu": empty_pairs,
            "tile_offsets_cpu": empty_offsets,
            "tile_keep_cpu": torch.empty((0,), dtype=torch.uint8),
            "stage1_pairs_cpu": empty_pairs.clone(),
            "tile_group_ids_cpu": empty_groups,
            "final_tile_is_tail_cpu": empty_tail,
            "stats": stats,
        }

    padded_height = ((matrix_height + K_vol - 1) // K_vol) * K_vol
    for stream in slot_streams:
        if len(stream) < padded_height:
            stream.extend([None] * (padded_height - len(stream)))

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pairs_list: list[list[list[int]]] = []
    tile_offsets_list: list[int] = []
    tile_group_ids_list: list[int] = []
    next_group_id = 0
    leftover_by_off: list[list[list[int]]] = [[] for _ in range(K_vol)]

    for row_idx in range(padded_height):
        row_offset = row_idx % K_vol
        row_pairs: list[list[int]] = []
        has_hole = False
        has_valid = False
        for slot_id in range(BM):
            entry = slot_streams[slot_id][row_idx]
            if entry is None:
                row_pairs.append([-1, -1])
                has_hole = True
            else:
                row_pairs.append(entry)
                has_valid = True
        if not has_valid:
            continue
        if not has_hole:
            tile_id = len(tile_pairs_list)
            tile_pairs_list.append(row_pairs)
            tile_offsets_list.append(row_offset)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1
        else:
            leftover_by_off[row_offset].extend([pair for pair in row_pairs if pair[0] >= 0])

    stage1_num_tiles = len(tile_pairs_list)

    for off in range(K_vol):
        leftover_pairs = leftover_by_off[off]
        for start in range(0, len(leftover_pairs), BM):
            chunk = leftover_pairs[start: start + BM]
            if not chunk:
                continue
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1
            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pair in enumerate(chunk):
                tile_pairs[slot] = pair[:]
            tile_pairs_list.append(tile_pairs)
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1

    valid_pairs = sum(
        1 for tile in tile_pairs_list for pair in tile if pair[0] >= 0
    )
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"slot-matrix direct schedule lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    debug = _build_schedule_from_tile_batches_cpu(
        tile_pairs_list,
        tile_offsets_list,
        tile_group_ids_list,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        pair_dtype,
        offset_dtype,
        1,
    )
    return debug


def _append_slot_matrix_position_cpu(
    slot_stream: list[tuple[int, int, int] | None],
    pos: int,
    residual_active_row: torch.Tensor,
    input_row_by_offset: torch.Tensor,
    out_row: int,
    K_vol: int,
    *,
    attach_mode: str,
) -> None:
    active_offsets = torch.nonzero(residual_active_row, as_tuple=False).view(-1).tolist()
    if not active_offsets:
        return

    current_phase = len(slot_stream) % K_vol
    if attach_mode == "phase_span":
        num_steps = max(((off - current_phase) % K_vol) for off in active_offsets) + 1
    elif attach_mode == "phase_block":
        num_steps = K_vol
    else:
        raise ValueError(f"unknown slot-matrix attach_mode: {attach_mode}")

    for delta in range(num_steps):
        off = (current_phase + delta) % K_vol
        input_row = int(input_row_by_offset[off].item())
        if input_row >= 0 and bool(residual_active_row[off].item()):
            slot_stream.append((pos, input_row, out_row))
        else:
            slot_stream.append(None)


def _assign_stage1_tile_ids_to_columns(
    stage1_tile_ids: list[int],
    num_blocks: int,
    *,
    mode: str,
    group_size: int,
) -> list[list[int]]:
    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    if not stage1_tile_ids or num_blocks <= 0:
        return columns

    if mode == "balance":
        column_lengths = [0] * num_blocks
        for tile_id in stage1_tile_ids:
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1
        return columns

    if mode == "chunked":
        total = len(stage1_tile_ids)
        base = total // num_blocks
        rem = total % num_blocks
        cursor = 0
        for col in range(num_blocks):
            count = base + (1 if col < rem else 0)
            if count > 0:
                columns[col].extend(stage1_tile_ids[cursor: cursor + count])
                cursor += count
        return columns

    if mode == "grouped_k":
        column_lengths = [0] * num_blocks
        step = max(group_size, 1)
        for start in range(0, len(stage1_tile_ids), step):
            chunk = stage1_tile_ids[start: start + step]
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].extend(chunk)
            column_lengths[target_col] += len(chunk)
        return columns

    raise ValueError(f"unknown slot-matrix stage1_column_mode: {mode}")


def _fill_slot_matrix_stage1_holes_cpu(
    tile_pairs_list: list[list[list[int]]],
    tile_offsets_list: list[int],
    columns: list[list[int]],
    residual_active: torch.Tensor,
    inputs_by_pos_and_offset: torch.Tensor,
    unique_out_rows: torch.Tensor,
    K_vol: int,
    *,
    fill_mode: str,
) -> int:
    if fill_mode == "none":
        return 0
    if fill_mode not in {"all", "isolated"}:
        raise ValueError(f"unknown slot-matrix hole fill mode: {fill_mode}")

    fill_order = _sort_positions_by_active_pattern(
        residual_active, unique_out_rows.tolist())
    fill_queues = [
        deque(pos for pos in fill_order if bool(residual_active[pos, off].item()))
        for off in range(K_vol)
    ]
    filled = 0

    if fill_mode == "all":
        tile_iter = (
            tile_id
            for column in columns
            for tile_id in column
        )
        for tile_id in tile_iter:
            off = tile_offsets_list[tile_id]
            tile_pairs = tile_pairs_list[tile_id]
            queue = fill_queues[off]
            for slot, pair in enumerate(tile_pairs):
                if pair[0] >= 0:
                    continue
                while queue and not bool(residual_active[queue[0], off].item()):
                    queue.popleft()
                if not queue:
                    break
                pos = queue.popleft()
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "slot-matrix hole fill encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(unique_out_rows[pos].item())]
                residual_active[pos, off] = False
                filled += 1
        return filled

    for column in columns:
        if not column:
            continue
        for col_row, tile_id in enumerate(column):
            prev_tile = column[col_row - 1] if col_row > 0 else -1
            next_tile = column[col_row + 1] if col_row + 1 < len(column) else -1
            prev_pairs = tile_pairs_list[prev_tile] if prev_tile >= 0 else None
            next_pairs = tile_pairs_list[next_tile] if next_tile >= 0 else None

            off = tile_offsets_list[tile_id]
            tile_pairs = tile_pairs_list[tile_id]
            queue = fill_queues[off]
            for slot, pair in enumerate(tile_pairs):
                if pair[0] >= 0:
                    continue
                prev_invalid = prev_pairs is None or prev_pairs[slot][0] < 0
                next_invalid = next_pairs is None or next_pairs[slot][0] < 0
                if not (prev_invalid and next_invalid):
                    continue
                while queue and not bool(residual_active[queue[0], off].item()):
                    queue.popleft()
                if not queue:
                    break
                pos = queue.popleft()
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "slot-matrix restricted hole fill encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(unique_out_rows[pos].item())]
                residual_active[pos, off] = False
                filled += 1
    return filled


def _build_slot_matrix_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    attach_mode: str,
    keep_threshold: int,
    num_rounds: int,
    stage1_column_mode: str,
    stage1_hole_fill_mode: str = "none",
) -> dict[str, torch.Tensor | dict[str, float]]:
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    residual_active = inputs_by_pos_and_offset.ge(0)

    tile_pairs_list: list[list[list[int]]] = []
    tile_offsets_list: list[int] = []
    tile_group_ids_list: list[int] = []
    stage1_tile_ids: list[int] = []
    next_group_id = 0
    completed_rounds = 0

    for _round in range(max(num_rounds, 1)):
        if not residual_active.any():
            break
        position_order = _sort_positions_by_active_pattern(
            residual_active, unique_out_rows.tolist())
        if not position_order:
            break

        slot_streams: list[list[tuple[int, int, int] | None]] = [[] for _ in range(BM)]
        for idx, pos in enumerate(position_order):
            slot_id = idx % BM
            _append_slot_matrix_position_cpu(
                slot_streams[slot_id],
                pos,
                residual_active[pos],
                inputs_by_pos_and_offset[pos],
                int(unique_out_rows[pos].item()),
                K_vol,
                attach_mode=attach_mode,
            )

        matrix_height = max((len(stream) for stream in slot_streams), default=0)
        if matrix_height <= 0:
            break
        padded_height = ((matrix_height + K_vol - 1) // K_vol) * K_vol
        for stream in slot_streams:
            if len(stream) < padded_height:
                stream.extend([None] * (padded_height - len(stream)))

        round_emitted = 0
        for row_idx in range(padded_height):
            row_offset = row_idx % K_vol
            row_pairs: list[list[int]] = []
            valid_positions: list[int] = []
            fill_count = 0
            for slot_id in range(BM):
                entry = slot_streams[slot_id][row_idx]
                if entry is None:
                    row_pairs.append([-1, -1])
                    continue
                pos, input_row, out_row = entry
                row_pairs.append([input_row, out_row])
                valid_positions.append(pos)
                fill_count += 1
            if fill_count <= 0 or fill_count < keep_threshold:
                continue

            tile_id = len(tile_pairs_list)
            tile_pairs_list.append(row_pairs)
            tile_offsets_list.append(row_offset)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1
            stage1_tile_ids.append(tile_id)
            round_emitted += 1
            for pos in valid_positions:
                residual_active[pos, row_offset] = False

        if round_emitted <= 0:
            break
        completed_rounds += 1

    stage1_num_tiles = len(tile_pairs_list)
    columns = _assign_stage1_tile_ids_to_columns(
        stage1_tile_ids,
        num_blocks,
        mode=stage1_column_mode,
        group_size=K_vol,
    )
    column_lengths = [len(col) for col in columns]
    refilled_pairs = 0
    if stage1_num_tiles > 0 and stage1_hole_fill_mode != "none":
        refilled_pairs = _fill_slot_matrix_stage1_holes_cpu(
            tile_pairs_list,
            tile_offsets_list,
            columns,
            residual_active,
            inputs_by_pos_and_offset,
            unique_out_rows,
            K_vol,
            fill_mode=stage1_hole_fill_mode,
        )

    tail_order = _sort_positions_by_active_pattern(
        residual_active, unique_out_rows.tolist())
    for off in range(K_vol):
        leftover_positions = [
            pos for pos in tail_order
            if bool(residual_active[pos, off].item())
        ]
        for start in range(0, len(leftover_positions), BM):
            chunk = leftover_positions[start: start + BM]
            if not chunk:
                continue
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1
            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pos in enumerate(chunk):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "slot-matrix tail encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(unique_out_rows[pos].item())]
                residual_active[pos, off] = False
            tile_pairs_list.append(tile_pairs)
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1

    if residual_active.any():
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"slot-matrix schedule left residual pairs after tail fill: {leftover_pairs}")

    debug = _build_schedule_from_tile_batches_cpu(
        tile_pairs_list,
        tile_offsets_list,
        tile_group_ids_list,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        pair_dtype,
        offset_dtype,
        completed_rounds,
    )
    debug["stats"] = dict(debug["stats"])
    debug["stats"]["stage1_refilled_pairs"] = float(refilled_pairs)
    debug["stats"]["stage1_hole_fill_mode"] = stage1_hole_fill_mode
    return debug


def _build_position_fill_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    order_mode: str,
    min_keep_support: int,
) -> dict[str, torch.Tensor | dict[str, float]]:
    device = triplets.device
    K_vol = int(offset_counts.numel())
    total_pairs = int(offset_counts.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets.dtype, device=device)
    inputs_by_pos_and_offset[inverse, triplets[:, 2].long()] = triplets[:, 0]
    active = inputs_by_pos_and_offset.ge(0)
    residual_active = active.clone()
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    masks = (active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    position_order = _position_fill_order_tensor(
        masks,
        active,
        unique_out_rows.to(dtype=torch.int64),
        order_mode=order_mode,
    )
    global_offset_support = active.sum(dim=0, dtype=torch.int64)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pair_batches: list[torch.Tensor] = []
    tile_offset_batches: list[torch.Tensor] = []
    tile_group_batches: list[torch.Tensor] = []
    total_tiles = 0
    next_group_id = 0

    num_full_groups = int(position_order.numel()) // BM
    for group_idx in range(num_full_groups):
        positions = position_order[group_idx * BM: (group_idx + 1) * BM]
        group_active = residual_active[positions]
        ordered_offsets = _ordered_offsets_by_group_support_tensor(
            group_active,
            global_offset_support,
        )
        if ordered_offsets.numel() == 0:
            continue
        group_support = group_active.sum(dim=0, dtype=torch.int64)
        keep_offsets = ordered_offsets[group_support[ordered_offsets] >= min_keep_support]
        if keep_offsets.numel() == 0:
            continue

        group_inputs = inputs_by_pos_and_offset[positions][:, keep_offsets]
        input_vals = group_inputs.transpose(0, 1).contiguous()
        row_vals = unique_out_rows[positions].to(pair_dtype).unsqueeze(0).expand(
            keep_offsets.numel(), -1)
        invalid_rows = torch.full_like(row_vals, -1)
        tile_pairs_batch = torch.empty(
            (int(keep_offsets.numel()), BM, 2), dtype=pair_dtype, device=device)
        tile_pairs_batch[:, :, 0] = input_vals.to(pair_dtype)
        tile_pairs_batch[:, :, 1] = torch.where(input_vals >= 0, row_vals, invalid_rows)
        tile_offsets_batch = keep_offsets.to(dtype=offset_dtype)
        tile_groups_batch = torch.full(
            (int(keep_offsets.numel()),), next_group_id, dtype=torch.int32, device=device)

        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)

        tile_base = total_tiles
        group_tile_ids = list(range(tile_base, tile_base + int(keep_offsets.numel())))
        target_col = min(range(num_blocks), key=column_lengths.__getitem__)
        columns[target_col].extend(group_tile_ids)
        column_lengths[target_col] += len(group_tile_ids)

        residual_active[positions.unsqueeze(1), keep_offsets.unsqueeze(0)] = False
        total_tiles += int(keep_offsets.numel())
        next_group_id += 1

    stage1_num_tiles = total_tiles

    for off in range(K_vol):
        leftover_positions = position_order[residual_active[position_order, off]]
        num_leftover = int(leftover_positions.numel())
        if num_leftover == 0:
            continue
        num_tail_tiles = (num_leftover + BM - 1) // BM
        slot_idx = torch.arange(num_leftover, device=device)
        tile_idx = slot_idx // BM
        tile_slot = slot_idx % BM

        tile_pairs_batch = torch.full(
            (num_tail_tiles, BM, 2), -1, dtype=pair_dtype, device=device)
        tile_pairs_batch[tile_idx, tile_slot, 0] = inputs_by_pos_and_offset[leftover_positions, off]
        tile_pairs_batch[tile_idx, tile_slot, 1] = unique_out_rows[leftover_positions].to(pair_dtype)
        tile_offsets_batch = torch.full(
            (num_tail_tiles,), off, dtype=offset_dtype, device=device)
        tile_groups_batch = torch.arange(
            next_group_id,
            next_group_id + num_tail_tiles,
            dtype=torch.int32,
            device=device,
        )

        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)

        tile_base = total_tiles
        for local_tile in range(num_tail_tiles):
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_base + local_tile)
            column_lengths[target_col] += 1

        residual_active[leftover_positions, off] = False
        total_tiles += num_tail_tiles
        next_group_id += num_tail_tiles

    if int(residual_active.sum().item()) != 0:
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"position-fill schedule left residual pairs after tail fill: {leftover_pairs}")

    return _finalize_schedule_from_tile_batches_gpu(
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        1,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        device=device,
        template_triplets=triplets,
        template_offsets=offset_counts,
    )


def _intersection_centrality_offset_order_tensor(
    residual_active: torch.Tensor,
) -> torch.Tensor:
    K_vol = int(residual_active.size(1))
    device = residual_active.device
    if K_vol == 0:
        return torch.empty((0,), dtype=torch.long, device=device)

    support = residual_active.sum(dim=0, dtype=torch.int64)
    centrality = torch.zeros((K_vol,), dtype=torch.int64, device=device)
    if K_vol > 1:
        pair_i, pair_j, pair_support = _compute_pair_support_gpu(residual_active)
        if pair_support.numel() > 0:
            pair_support_i64 = pair_support.to(torch.int64)
            centrality.scatter_add_(0, pair_i, pair_support_i64)
            centrality.scatter_add_(0, pair_j, pair_support_i64)

    order = torch.arange(K_vol, device=device, dtype=torch.long)
    if K_vol > 1:
        order = order[torch.argsort(support[order], descending=True, stable=True)]
        order = order[torch.argsort(centrality[order], descending=True, stable=True)]
    return order


def _centrality_signature_order_tensor(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    out_rows: torch.Tensor,
    *,
    prefix_len: int | None = None,
) -> torch.Tensor:
    K_vol = int(residual_active.size(1))
    active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1)
    if active_positions.numel() == 0:
        return active_positions

    device = residual_masks.device
    offset_order = _intersection_centrality_offset_order_tensor(residual_active)
    active_ranked = residual_active[active_positions][:, offset_order]
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    reordered_masks = (active_ranked.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)

    row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1
    mask_base = 1 << K_vol
    popcount = residual_active[active_positions].sum(dim=1, dtype=torch.int64)
    refine_key = (
        ((K_vol + 1 - popcount) * mask_base) + reordered_masks
    ) * row_base + out_rows[active_positions]

    if prefix_len is None:
        order = torch.argsort(refine_key)
        return active_positions[order]

    prefix_base = K_vol + 1
    rank_positions = torch.arange(K_vol, dtype=torch.int64, device=device).unsqueeze(0)
    masked_rank_positions = torch.where(
        active_ranked,
        rank_positions.expand(active_ranked.size(0), -1),
        torch.full((active_ranked.size(0), K_vol), K_vol, dtype=torch.int64, device=device),
    )
    topk = min(prefix_len, K_vol)
    prefix_tokens = torch.topk(masked_rank_positions, k=topk, dim=1, largest=False).values
    if topk < prefix_len:
        pad = torch.full(
            (prefix_tokens.size(0), prefix_len - topk),
            K_vol,
            dtype=torch.int64,
            device=device,
        )
        prefix_tokens = torch.cat((prefix_tokens, pad), dim=1)

    prefix_signature = prefix_tokens[:, 0]
    for idx in range(1, prefix_len):
        prefix_signature = prefix_signature * prefix_base + prefix_tokens[:, idx]

    refine_base = int(refine_key.max().item()) + 1 if refine_key.numel() > 0 else 1
    combined_key = prefix_signature * refine_base + refine_key
    order = torch.argsort(combined_key)
    return active_positions[order]


def _finalize_schedule_from_tile_batches_gpu(
    tile_pair_batches: list[torch.Tensor],
    tile_offset_batches: list[torch.Tensor],
    tile_group_batches: list[torch.Tensor],
    columns: list[list[int]],
    stage1_num_tiles: int,
    total_pairs: int,
    num_positions: int,
    BM: int,
    num_blocks: int,
    num_rounds: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    device: torch.device,
    template_triplets: torch.Tensor,
    template_offsets: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, float]]:
    if not tile_pair_batches:
        empty_pairs = template_triplets.new_empty((0, 2))
        empty_offsets = template_offsets.new_empty((0,))
        empty_groups = template_offsets.new_empty((0,), dtype=torch.int32)
        empty_tail = torch.empty((0,), dtype=torch.bool, device=device)
        stats = {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": float(num_positions),
            "hole_count": 0.0,
            "stage1_hole_count": 0.0,
            "stage1_fill_ratio": 0.0,
            "stage1_num_tiles": 0.0,
            "tail_num_tiles": 0.0,
            "num_rounds": float(num_rounds),
        }
        return {
            "scheduled_pairs_cpu": empty_pairs,
            "tile_offsets_cpu": empty_offsets,
            "tile_keep_cpu": torch.empty((0,), dtype=torch.uint8, device=device),
            "stage1_pairs_cpu": empty_pairs.clone(),
            "tile_group_ids_cpu": empty_groups,
            "final_tile_is_tail_cpu": empty_tail,
            "stats": stats,
        }

    all_tile_pairs = torch.cat(tile_pair_batches, dim=0)
    all_tile_offsets = torch.cat(tile_offset_batches, dim=0)
    all_tile_groups = torch.cat(tile_group_batches, dim=0)
    stage1_columns = [
        [tile_id for tile_id in col if tile_id < stage1_num_tiles]
        for col in columns
    ]
    stage1_flat_tile_ids = _flatten_schedule_columns_row_major(stage1_columns)
    stage1_indices = torch.tensor(
        stage1_flat_tile_ids, dtype=torch.long, device=device)
    final_flat_tile_ids = _flatten_schedule_columns_row_major(columns)
    final_indices = torch.tensor(
        final_flat_tile_ids, dtype=torch.long, device=device)
    stage1_pairs = (
        all_tile_pairs[stage1_indices].reshape(-1, 2)
        if stage1_indices.numel() > 0
        else template_triplets.new_empty((0, 2))
    )
    scheduled_pairs = all_tile_pairs[final_indices].reshape(-1, 2)
    tile_offsets = all_tile_offsets[final_indices]
    tile_group_ids = all_tile_groups[final_indices]
    final_tile_is_tail = final_indices >= stage1_num_tiles
    tile_keep = _compute_tile_keep_gpu(
        scheduled_pairs,
        final_tile_is_tail,
        BM,
        num_blocks,
    )

    stage1_valid_pairs = int((stage1_pairs[:, 0] >= 0).sum().item())
    stage1_hole_count = int(stage1_pairs.size(0) - stage1_valid_pairs)
    valid_pairs = int((scheduled_pairs[:, 0] >= 0).sum().item())
    hole_count = int(scheduled_pairs.size(0) - valid_pairs)
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"minimal scheduled worklist lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    stats = {
        "num_tiles": float(tile_offsets.numel()),
        "num_blocks": float(num_blocks),
        "padding_ratio": float(scheduled_pairs.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(scheduled_pairs.size(0), 1)),
        "num_positions": float(num_positions),
        "hole_count": float(hole_count),
        "stage1_hole_count": float(stage1_hole_count),
        "stage1_fill_ratio": float(stage1_valid_pairs / max(stage1_pairs.size(0), 1)),
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(all_tile_offsets.numel() - stage1_num_tiles),
        "num_rounds": float(num_rounds),
    }
    return {
        "scheduled_pairs_cpu": scheduled_pairs,
        "tile_offsets_cpu": tile_offsets,
        "tile_keep_cpu": tile_keep,
        "stage1_pairs_cpu": stage1_pairs,
        "tile_group_ids_cpu": tile_group_ids,
        "final_tile_is_tail_cpu": final_tile_is_tail,
        "stats": stats,
    }


def _materialize_exact_group_tiles_gpu(
    group_positions: torch.Tensor,
    group_masks: torch.Tensor,
    group_ids: torch.Tensor,
    inputs_by_pos_and_offset: torch.Tensor,
    out_rows: torch.Tensor,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    bit_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_positions.numel() == 0:
        empty_pairs = torch.empty((0, int(inputs_by_pos_and_offset.size(1)), 2), dtype=pair_dtype, device=inputs_by_pos_and_offset.device)
        empty_offsets = torch.empty((0,), dtype=offset_dtype, device=inputs_by_pos_and_offset.device)
        empty_groups = torch.empty((0,), dtype=torch.int32, device=inputs_by_pos_and_offset.device)
        empty_counts = torch.empty((0,), dtype=torch.int32, device=inputs_by_pos_and_offset.device)
        return empty_pairs, empty_offsets, empty_groups, empty_counts

    membership = (group_masks.unsqueeze(1) & bit_values.unsqueeze(0)) != 0
    tile_counts = membership.sum(dim=1, dtype=torch.int32)
    members = torch.nonzero(membership, as_tuple=False)
    if members.numel() == 0:
        empty_pairs = torch.empty((0, int(group_positions.size(1)), 2), dtype=pair_dtype, device=inputs_by_pos_and_offset.device)
        empty_offsets = torch.empty((0,), dtype=offset_dtype, device=inputs_by_pos_and_offset.device)
        empty_groups = torch.empty((0,), dtype=torch.int32, device=inputs_by_pos_and_offset.device)
        return empty_pairs, empty_offsets, empty_groups, tile_counts
    group_sel = members[:, 0]
    off_sel = members[:, 1]
    group_pos_rep = group_positions[group_sel]
    input_rows = inputs_by_pos_and_offset[group_pos_rep, off_sel.unsqueeze(1)]
    out_rows_batch = out_rows[group_pos_rep].to(pair_dtype)
    tile_pairs = torch.stack((input_rows, out_rows_batch), dim=-1).to(pair_dtype)
    tile_offsets = off_sel.to(dtype=offset_dtype)
    tile_groups = group_ids[group_sel].to(dtype=torch.int32)
    return tile_pairs, tile_offsets, tile_groups, tile_counts


def _build_single_round_exact_group_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    order_mode: str,
) -> dict[str, torch.Tensor | dict[str, float]]:
    device = triplets.device
    K_vol = int(offset_counts.numel())
    total_pairs = int(offset_counts.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets.dtype, device=device)
    inputs_by_pos_and_offset[inverse, triplets[:, 2].long()] = triplets[:, 0]
    out_rows = unique_out_rows.to(dtype=torch.int64)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    residual_masks = (
        (inputs_by_pos_and_offset.ge(0).to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    )
    residual_active = inputs_by_pos_and_offset.ge(0)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pair_batches: list[torch.Tensor] = []
    tile_offset_batches: list[torch.Tensor] = []
    tile_group_batches: list[torch.Tensor] = []
    total_tiles = 0
    next_group_id = 0
    num_rounds = 1

    ordered_positions = _single_round_exact_group_order_tensor(
        residual_masks,
        residual_active,
        out_rows,
        order_mode=order_mode,
    )
    num_groups = int(ordered_positions.numel()) // BM
    if num_groups > 0:
        group_positions = ordered_positions[: num_groups * BM].view(num_groups, BM)
        group_masks = residual_masks[group_positions[:, 0]].clone()
        for slot in range(1, BM):
            group_masks &= residual_masks[group_positions[:, slot]]
        group_popcount = ((group_masks.unsqueeze(1) & bit_values.unsqueeze(0)) != 0).sum(dim=1)
        valid = group_popcount >= 2
        if bool(valid.any().item()):
            valid_group_positions = group_positions[valid]
            valid_group_masks = group_masks[valid]
            group_ids = torch.arange(
                next_group_id,
                next_group_id + int(valid_group_positions.size(0)),
                dtype=torch.int32,
                device=device,
            )
            tile_pairs, tile_offsets, tile_groups, tile_counts = _materialize_exact_group_tiles_gpu(
                valid_group_positions,
                valid_group_masks,
                group_ids,
                inputs_by_pos_and_offset,
                out_rows,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
                bit_values=bit_values,
            )
            if tile_offsets.numel() > 0:
                tile_pair_batches.append(tile_pairs)
                tile_offset_batches.append(tile_offsets)
                tile_group_batches.append(tile_groups)
                tile_base = total_tiles
                for tile_count in tile_counts.tolist():
                    target_col = min(range(num_blocks), key=column_lengths.__getitem__)
                    group_tile_ids = list(range(tile_base, tile_base + int(tile_count)))
                    columns[target_col].extend(group_tile_ids)
                    column_lengths[target_col] += int(tile_count)
                    tile_base += int(tile_count)
                total_tiles += int(tile_counts.sum().item())
                next_group_id += int(valid_group_positions.size(0))
                residual_masks[valid_group_positions] &= ~valid_group_masks.unsqueeze(1)
                residual_active = (residual_masks.unsqueeze(1) & bit_values.unsqueeze(0)) != 0

    stage1_num_tiles = total_tiles

    tail_order = _single_round_exact_group_order_tensor(
        residual_masks,
        residual_active,
        out_rows,
        order_mode=order_mode,
    )
    for off in range(K_vol):
        leftover_positions = tail_order[residual_active[tail_order, off]]
        num_leftover = int(leftover_positions.numel())
        if num_leftover == 0:
            continue
        num_tail_tiles = (num_leftover + BM - 1) // BM
        slot_idx = torch.arange(num_leftover, device=device)
        tile_idx = slot_idx // BM
        tile_slot = slot_idx % BM
        tile_pairs_batch = torch.full(
            (num_tail_tiles, BM, 2), -1, dtype=pair_dtype, device=device)
        tile_pairs_batch[tile_idx, tile_slot, 0] = inputs_by_pos_and_offset[leftover_positions, off]
        tile_pairs_batch[tile_idx, tile_slot, 1] = out_rows[leftover_positions].to(pair_dtype)
        tile_offsets_batch = torch.full(
            (num_tail_tiles,), off, dtype=offset_dtype, device=device)
        tile_groups_batch = torch.arange(
            next_group_id,
            next_group_id + num_tail_tiles,
            dtype=torch.int32,
            device=device,
        )
        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)
        tile_base = total_tiles
        for local_tile in range(num_tail_tiles):
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_base + local_tile)
            column_lengths[target_col] += 1
        residual_masks[leftover_positions] &= ~bit_values[off]
        residual_active[leftover_positions, off] = False
        total_tiles += num_tail_tiles
        next_group_id += num_tail_tiles

    if int(residual_active.sum().item()) != 0:
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"support-sorted schedule left residual pairs after tail fill: {leftover_pairs}")

    return _finalize_schedule_from_tile_batches_gpu(
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        num_rounds,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        device=device,
        template_triplets=triplets,
        template_offsets=offset_counts,
    )


def _build_single_round_exact_group_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    order_mode: str,
) -> dict[str, torch.Tensor | dict[str, float]]:
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    residual_active = inputs_by_pos_and_offset.ge(0)
    out_rows_cpu = unique_out_rows.to(dtype=torch.int64)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64))
    residual_masks = (residual_active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pairs_list: list[list[list[int]]] = []
    tile_offsets_list: list[int] = []
    tile_group_ids_list: list[int] = []
    next_group_id = 0
    num_rounds = 1

    ordered_positions = _single_round_exact_group_order_cpu(
        residual_masks,
        residual_active,
        out_rows_cpu,
        order_mode=order_mode,
    )
    for group_start in range(0, len(ordered_positions), BM):
        group_positions = ordered_positions[group_start: group_start + BM]
        if len(group_positions) != BM:
            break
        group_mask = int(residual_masks[group_positions[0]].item())
        for pos in group_positions[1:]:
            group_mask &= int(residual_masks[pos].item())
        if group_mask == 0 or len(_mask_to_offsets(group_mask, K_vol)) <= 1:
            continue
        next_group_id = _emit_exact_group_tiles_cpu(
            group_positions,
            group_mask,
            inputs_by_pos_and_offset,
            out_rows_cpu,
            tile_pairs_list,
            tile_offsets_list,
            tile_group_ids_list,
            columns,
            column_lengths,
            next_group_id,
        )
        for off in _mask_to_offsets(group_mask, K_vol):
            residual_active[group_positions, off] = False
            residual_masks[group_positions] -= (1 << off)

    stage1_num_tiles = len(tile_pairs_list)

    tail_order = _single_round_exact_group_order_cpu(
        residual_masks,
        residual_active,
        out_rows_cpu,
        order_mode=order_mode,
    )
    for off in range(K_vol):
        leftover_positions = [pos for pos in tail_order if bool(residual_active[pos, off].item())]
        for start in range(0, len(leftover_positions), BM):
            chunk = leftover_positions[start: start + BM]
            if not chunk:
                continue
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1
            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pos in enumerate(chunk):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "support-sorted tail encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(out_rows_cpu[pos].item())]
                residual_active[pos, off] = False
                residual_masks[pos] -= (1 << off)
            tile_pairs_list.append(tile_pairs)
            target_col = min(range(num_blocks), key=lambda col: column_lengths[col])
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1

    if residual_active.any():
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"support-sorted schedule left residual pairs after tail fill: {leftover_pairs}")

    return _build_schedule_from_tile_batches_cpu(
        tile_pairs_list,
        tile_offsets_list,
        tile_group_ids_list,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        pair_dtype,
        offset_dtype,
        num_rounds,
    )


def _build_support_sorted_mask_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="support_sorted_mask",
    )


def _build_support_sorted_mask_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_cpu_debug(
        triplets_cpu,
        offset_counts_cpu,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="support_sorted_mask",
    )


def _build_dplane_popcount_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="dplane_popcount",
    )


def _build_dplane_popcount_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_cpu_debug(
        triplets_cpu,
        offset_counts_cpu,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="dplane_popcount",
    )


def _build_dplane_support_sorted_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="dplane_support_sorted",
    )


def _build_dplane_support_sorted_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_cpu_debug(
        triplets_cpu,
        offset_counts_cpu,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="dplane_support_sorted",
    )


def _build_axis_marginal_support_sorted_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="axis_marginal_support_sorted",
    )


def _build_axis_marginal_support_sorted_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_single_round_exact_group_schedule_cpu_debug(
        triplets_cpu,
        offset_counts_cpu,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        order_mode="axis_marginal_support_sorted",
    )


def _build_ordered_signature_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    num_rounds: int,
    prefix_len: int | None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    device = triplets.device
    K_vol = int(offset_counts.numel())
    total_pairs = int(offset_counts.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets.dtype, device=device)
    inputs_by_pos_and_offset[inverse, triplets[:, 2].long()] = triplets[:, 0]
    out_rows = unique_out_rows.to(dtype=torch.int64)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    residual_masks = (
        (inputs_by_pos_and_offset.ge(0).to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    )
    residual_active = inputs_by_pos_and_offset.ge(0)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pair_batches: list[torch.Tensor] = []
    tile_offset_batches: list[torch.Tensor] = []
    tile_group_batches: list[torch.Tensor] = []
    total_tiles = 0
    next_group_id = 0
    completed_rounds = 0

    for _ in range(num_rounds):
        ordered_positions = _centrality_signature_order_tensor(
            residual_masks,
            residual_active,
            out_rows,
            prefix_len=prefix_len,
        )
        num_groups = int(ordered_positions.numel()) // BM
        if num_groups <= 0:
            break

        group_positions = ordered_positions[: num_groups * BM].view(num_groups, BM)
        group_masks = residual_masks[group_positions[:, 0]].clone()
        for slot in range(1, BM):
            group_masks &= residual_masks[group_positions[:, slot]]

        group_popcount = ((group_masks.unsqueeze(1) & bit_values.unsqueeze(0)) != 0).sum(dim=1)
        valid = group_popcount >= 2
        if not bool(valid.any().item()):
            break

        valid_group_positions = group_positions[valid]
        valid_group_masks = group_masks[valid]
        group_ids = torch.arange(
            next_group_id,
            next_group_id + int(valid_group_positions.size(0)),
            dtype=torch.int32,
            device=device,
        )
        tile_pairs, tile_offsets, tile_groups, tile_counts = _materialize_exact_group_tiles_gpu(
            valid_group_positions,
            valid_group_masks,
            group_ids,
            inputs_by_pos_and_offset,
            out_rows,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            bit_values=bit_values,
        )
        if tile_offsets.numel() == 0:
            break

        tile_pair_batches.append(tile_pairs)
        tile_offset_batches.append(tile_offsets)
        tile_group_batches.append(tile_groups)
        tile_base = total_tiles
        for tile_count in tile_counts.tolist():
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            group_tile_ids = list(range(tile_base, tile_base + int(tile_count)))
            columns[target_col].extend(group_tile_ids)
            column_lengths[target_col] += int(tile_count)
            tile_base += int(tile_count)

        total_tiles += int(tile_counts.sum().item())
        next_group_id += int(valid_group_positions.size(0))
        residual_masks[valid_group_positions] &= ~valid_group_masks.unsqueeze(1)
        updated_masks = residual_masks[valid_group_positions]
        residual_active[valid_group_positions] = (
            updated_masks.unsqueeze(-1) & bit_values.view(1, 1, -1)
        ) != 0
        completed_rounds += 1

    stage1_num_tiles = total_tiles

    tail_order = _centrality_signature_order_tensor(
        residual_masks,
        residual_active,
        out_rows,
        prefix_len=prefix_len,
    )
    for off in range(K_vol):
        leftover_positions = tail_order[residual_active[tail_order, off]]
        num_leftover = int(leftover_positions.numel())
        if num_leftover == 0:
            continue
        num_tail_tiles = (num_leftover + BM - 1) // BM
        slot_idx = torch.arange(num_leftover, device=device)
        tile_idx = slot_idx // BM
        tile_slot = slot_idx % BM
        tile_pairs_batch = torch.full(
            (num_tail_tiles, BM, 2), -1, dtype=pair_dtype, device=device)
        tile_pairs_batch[tile_idx, tile_slot, 0] = inputs_by_pos_and_offset[leftover_positions, off]
        tile_pairs_batch[tile_idx, tile_slot, 1] = out_rows[leftover_positions].to(pair_dtype)
        tile_offsets_batch = torch.full(
            (num_tail_tiles,), off, dtype=offset_dtype, device=device)
        tile_groups_batch = torch.arange(
            next_group_id,
            next_group_id + num_tail_tiles,
            dtype=torch.int32,
            device=device,
        )
        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)
        tile_base = total_tiles
        for local_tile in range(num_tail_tiles):
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_base + local_tile)
            column_lengths[target_col] += 1
        residual_masks[leftover_positions] &= ~bit_values[off]
        residual_active[leftover_positions, off] = False
        total_tiles += num_tail_tiles
        next_group_id += num_tail_tiles

    if int(residual_active.sum().item()) != 0:
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"ordered-signature schedule left residual pairs after tail fill: {leftover_pairs}")

    return _finalize_schedule_from_tile_batches_gpu(
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        completed_rounds,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        device=device,
        template_triplets=triplets,
        template_offsets=offset_counts,
    )


def _build_ordered_signature_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    num_rounds: int,
    prefix_len: int | None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    residual_active = inputs_by_pos_and_offset.ge(0)
    out_rows_cpu = unique_out_rows.to(dtype=torch.int64)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64))
    residual_masks = (residual_active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pairs_list: list[list[list[int]]] = []
    tile_offsets_list: list[int] = []
    tile_group_ids_list: list[int] = []
    next_group_id = 0
    completed_rounds = 0

    for _ in range(num_rounds):
        ordered_positions = _centrality_signature_order_tensor(
            residual_masks,
            residual_active,
            out_rows_cpu,
            prefix_len=prefix_len,
        ).tolist()
        round_emitted = False
        for group_start in range(0, len(ordered_positions), BM):
            group_positions = ordered_positions[group_start: group_start + BM]
            if len(group_positions) != BM:
                break
            group_mask = int(residual_masks[group_positions[0]].item())
            for pos in group_positions[1:]:
                group_mask &= int(residual_masks[pos].item())
            if group_mask == 0 or len(_mask_to_offsets(group_mask, K_vol)) <= 1:
                continue
            next_group_id = _emit_exact_group_tiles_cpu(
                group_positions,
                group_mask,
                inputs_by_pos_and_offset,
                out_rows_cpu,
                tile_pairs_list,
                tile_offsets_list,
                tile_group_ids_list,
                columns,
                column_lengths,
                next_group_id,
            )
            for off in _mask_to_offsets(group_mask, K_vol):
                residual_active[group_positions, off] = False
                residual_masks[group_positions] -= (1 << off)
            round_emitted = True
        if not round_emitted:
            break
        completed_rounds += 1

    stage1_num_tiles = len(tile_pairs_list)

    tail_order = _centrality_signature_order_tensor(
        residual_masks,
        residual_active,
        out_rows_cpu,
        prefix_len=prefix_len,
    ).tolist()
    for off in range(K_vol):
        leftover_positions = [pos for pos in tail_order if bool(residual_active[pos, off].item())]
        for start in range(0, len(leftover_positions), BM):
            chunk = leftover_positions[start: start + BM]
            if not chunk:
                continue
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1
            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pos in enumerate(chunk):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "ordered-signature tail encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(out_rows_cpu[pos].item())]
                residual_active[pos, off] = False
                residual_masks[pos] -= (1 << off)
            tile_pairs_list.append(tile_pairs)
            target_col = min(range(num_blocks), key=lambda col: column_lengths[col])
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1

    if residual_active.any():
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"ordered-signature schedule left residual pairs after tail fill: {leftover_pairs}")

    return _build_schedule_from_tile_batches_cpu(
        tile_pairs_list,
        tile_offsets_list,
        tile_group_ids_list,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        pair_dtype,
        offset_dtype,
        completed_rounds,
    )


def _run_subset_sweep_schedule_gpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    residual_popcount: torch.Tensor,
    inputs_by_pos_and_offset: torch.Tensor,
    out_rows: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    subset_size_sequence: tuple[int, ...],
    max_candidates_sequence: tuple[int | None, ...] | None = None,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    bit_values: torch.Tensor,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[int]],
    list[int],
    int,
    int,
    int,
]:
    device = residual_masks.device
    K_vol = int(inputs_by_pos_and_offset.size(1))
    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pair_batches: list[torch.Tensor] = []
    tile_offset_batches: list[torch.Tensor] = []
    tile_group_batches: list[torch.Tensor] = []
    total_tiles = 0
    next_group_id = 0
    completed_rounds = 0
    mask_base = 1 << K_vol
    row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1

    if max_candidates_sequence is None:
        max_candidates_sequence = (None,) * len(subset_size_sequence)
    if len(max_candidates_sequence) != len(subset_size_sequence):
        raise ValueError("max_candidates_sequence must align with subset_size_sequence")

    for pass_idx, subset_size in enumerate(subset_size_sequence):
        offset_support = residual_active.sum(dim=0, dtype=torch.int32)
        candidate_offset_mask = int(
            ((offset_support >= BM).to(torch.int64) * bit_values).sum().item()
        )
        if candidate_offset_mask == 0:
            continue

        subset_masks_bank, subset_sizes_bank = _get_subset_mask_bank(
            K_vol, subset_size, device)
        valid_bank = (
            (subset_sizes_bank.to(torch.int32) == subset_size) &
            ((subset_masks_bank & candidate_offset_mask) == subset_masks_bank)
        )
        subset_masks = subset_masks_bank[valid_bank]
        if subset_masks.numel() == 0:
            continue

        active_positions = torch.nonzero(residual_popcount > 0, as_tuple=False).view(-1)
        if int(active_positions.numel()) < BM:
            break

        support_scores: list[torch.Tensor] = []
        chunk_size = 1024
        active_masks = residual_masks[active_positions]
        for start in range(0, int(subset_masks.numel()), chunk_size):
            chunk_masks = subset_masks[start: start + chunk_size]
            if chunk_masks.numel() == 0:
                continue
            support_counts = (
                ((active_masks.unsqueeze(1) & chunk_masks.unsqueeze(0)) == chunk_masks.unsqueeze(0))
                .sum(dim=0, dtype=torch.int32)
            )
            num_groups = support_counts // BM
            score = num_groups.to(torch.int64) * (int(active_positions.numel()) + 1) + support_counts.to(torch.int64)
            support_scores.append(score)
        if not support_scores:
            continue
        support_scores_t = torch.cat(support_scores, dim=0)
        valid_subset = support_scores_t >= (BM + 1)
        if not bool(valid_subset.any().item()):
            continue

        candidate_masks = subset_masks[valid_subset]
        candidate_scores = support_scores_t[valid_subset]
        order = torch.argsort(candidate_scores, descending=True, stable=True)
        max_candidates_per_pass = max_candidates_sequence[pass_idx]
        if max_candidates_per_pass is not None:
            order = order[:max_candidates_per_pass]
        ordered_masks = candidate_masks[order]

        pass_emitted = False
        for subset_mask in ordered_masks.tolist():
            support_positions = torch.nonzero(
                (residual_masks & subset_mask) == subset_mask,
                as_tuple=False,
            ).view(-1)
            if int(support_positions.numel()) < BM:
                continue

            support_keys = (
                (residual_popcount[support_positions].to(torch.int64) * mask_base + residual_masks[support_positions])
                * row_base
                + out_rows[support_positions]
            )
            support_order = torch.argsort(support_keys)
            support_positions = support_positions[support_order]
            num_groups = int(support_positions.numel()) // BM
            if num_groups == 0:
                continue

            subset_offsets = _mask_to_offsets(int(subset_mask), K_vol)
            if len(subset_offsets) != subset_size:
                continue

            group_positions = support_positions[: num_groups * BM].view(num_groups, BM)
            group_out_rows = out_rows[group_positions]
            group_inputs = torch.stack(
                [inputs_by_pos_and_offset[group_positions, off] for off in subset_offsets],
                dim=1,
            )
            tile_pairs_batch = torch.stack(
                (
                    group_inputs,
                    group_out_rows.unsqueeze(1).expand(-1, subset_size, -1),
                ),
                dim=-1,
            ).reshape(-1, BM, 2).to(dtype=pair_dtype)
            tile_offsets_batch = torch.tensor(
                subset_offsets, dtype=offset_dtype, device=device).repeat(num_groups)
            tile_groups_batch = torch.arange(
                next_group_id,
                next_group_id + num_groups,
                dtype=torch.int32,
                device=device,
            ).repeat_interleave(subset_size)

            tile_pair_batches.append(tile_pairs_batch)
            tile_offset_batches.append(tile_offsets_batch)
            tile_group_batches.append(tile_groups_batch)

            tile_base = total_tiles
            for group_idx in range(num_groups):
                target_col = min(range(num_blocks), key=column_lengths.__getitem__)
                group_tile_ids = list(range(
                    tile_base + group_idx * subset_size,
                    tile_base + (group_idx + 1) * subset_size,
                ))
                columns[target_col].extend(group_tile_ids)
                column_lengths[target_col] += subset_size

            group_flat_positions = group_positions.reshape(-1)
            for off in subset_offsets:
                bit = 1 << off
                residual_active[group_positions, off] = False
                residual_masks[group_flat_positions] -= bit
                residual_popcount[group_flat_positions] -= 1

            total_tiles += num_groups * subset_size
            next_group_id += num_groups
            pass_emitted = True

        if pass_emitted:
            completed_rounds += 1

    return (
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        column_lengths,
        total_tiles,
        next_group_id,
        completed_rounds,
    )


def _build_subset_sweep_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    subset_size_sequence: tuple[int, ...],
    max_candidates_sequence: tuple[int | None, ...] | None,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    device = triplets.device
    K_vol = int(offset_counts.numel())
    total_pairs = int(offset_counts.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets.dtype, device=device)
    inputs_by_pos_and_offset[inverse, triplets[:, 2].long()] = triplets[:, 0]

    active = inputs_by_pos_and_offset.ge(0)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    residual_active = active.clone()
    residual_masks = (active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    residual_popcount = active.sum(dim=1, dtype=torch.int32)
    out_rows_tensor = unique_out_rows.to(dtype=torch.int64)
    base_sorted_positions = _sort_positions_by_mask_key_tensor(
        residual_masks, residual_popcount, out_rows_tensor, K_vol)

    (
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        column_lengths,
        total_tiles,
        next_group_id,
        num_rounds,
    ) = _run_subset_sweep_schedule_gpu(
        residual_masks,
        residual_active,
        residual_popcount,
        inputs_by_pos_and_offset,
        out_rows_tensor,
        BM,
        num_blocks,
        subset_size_sequence=subset_size_sequence,
        max_candidates_sequence=max_candidates_sequence,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        bit_values=bit_values,
    )

    stage1_num_tiles = total_tiles
    for off in range(K_vol):
        leftover_positions = base_sorted_positions[residual_active[base_sorted_positions, off]]
        num_leftover = int(leftover_positions.numel())
        if num_leftover == 0:
            continue
        num_tail_tiles = (num_leftover + BM - 1) // BM
        slot_idx = torch.arange(num_leftover, device=device)
        tile_idx = slot_idx // BM
        tile_slot = slot_idx % BM

        tile_pairs_batch = torch.full(
            (num_tail_tiles, BM, 2), -1, dtype=pair_dtype, device=device)
        tile_pairs_batch[tile_idx, tile_slot, 0] = inputs_by_pos_and_offset[leftover_positions, off]
        tile_pairs_batch[tile_idx, tile_slot, 1] = out_rows_tensor[leftover_positions].to(pair_dtype)
        tile_offsets_batch = torch.full(
            (num_tail_tiles,), off, dtype=offset_dtype, device=device)
        tile_groups_batch = torch.arange(
            next_group_id,
            next_group_id + num_tail_tiles,
            dtype=torch.int32,
            device=device,
        )

        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)

        tile_base = total_tiles
        for local_tile in range(num_tail_tiles):
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_base + local_tile)
            column_lengths[target_col] += 1

        bit = 1 << off
        residual_active[leftover_positions, off] = False
        residual_masks[leftover_positions] -= bit
        residual_popcount[leftover_positions] -= 1
        total_tiles += num_tail_tiles
        next_group_id += num_tail_tiles

    if int(residual_popcount.sum().item()) != 0:
        leftover_pairs = int(residual_popcount.sum().item())
        raise RuntimeError(
            f"subset sweep schedule left residual pairs after tail fill: {leftover_pairs}")

    return _finalize_schedule_from_tile_batches_gpu(
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        num_rounds,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        device=device,
        template_triplets=triplets,
        template_offsets=offset_counts,
    )


def _compute_pair_support_cpu(
    residual_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    K_vol = int(residual_active.size(1))
    pair_i: list[int] = []
    pair_j: list[int] = []
    for i in range(K_vol):
        for j in range(i + 1, K_vol):
            pair_i.append(i)
            pair_j.append(j)
    if not pair_i:
        return (
            torch.empty((0,), dtype=torch.int16),
            torch.empty((0,), dtype=torch.int16),
            torch.empty((0,), dtype=torch.int32),
        )
    pair_i_t = torch.tensor(pair_i, dtype=torch.int16)
    pair_j_t = torch.tensor(pair_j, dtype=torch.int16)
    pair_support = (
        residual_active[:, pair_i_t.long()] & residual_active[:, pair_j_t.long()]
    ).sum(dim=0, dtype=torch.int32)
    return pair_i_t, pair_j_t, pair_support


def _compute_pair_support_gpu(
    residual_active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    K_vol = int(residual_active.size(1))
    if K_vol <= 1:
        empty = torch.empty((0,), dtype=torch.int64, device=residual_active.device)
        return empty, empty, torch.empty((0,), dtype=torch.int32, device=residual_active.device)
    pair_i, pair_j = torch.triu_indices(K_vol, K_vol, offset=1, device=residual_active.device)
    pair_support = (
        residual_active[:, pair_i] & residual_active[:, pair_j]
    ).sum(dim=0, dtype=torch.int32)
    return pair_i, pair_j, pair_support


def _compute_sig2_pair_score_gpu(
    pair_i_valid: torch.Tensor,
    pair_j_valid: torch.Tensor,
    support_valid: torch.Tensor,
    support_per_offset: torch.Tensor,
    active_count: int,
    BM: int,
    *,
    score_mode: str,
) -> torch.Tensor:
    support_valid = support_valid.to(torch.int64)
    support_per_offset = support_per_offset.to(torch.int64)

    if score_mode == "min_support":
        tie_valid = support_per_offset[pair_i_valid] + support_per_offset[pair_j_valid]
        return ((support_valid * (support_per_offset.numel() ** 2)) +
                (tie_valid * support_per_offset.numel()) +
                pair_i_valid.to(torch.int64)) * support_per_offset.numel() + pair_j_valid.to(torch.int64)

    if score_mode in {
        "target_2x",
        "target_4x",
        "target_2x_balance",
        "target_4x_balance",
    }:
        target_mult = 2 if "2x" in score_mode else 4
        target_support = max(BM, min(active_count, target_mult * BM))
        support_gap = torch.abs(support_valid - target_support)

        if "balance" in score_mode:
            support_i = support_per_offset[pair_i_valid]
            support_j = support_per_offset[pair_j_valid]
            n11 = support_valid
            n10 = support_i - n11
            n01 = support_j - n11
            n00 = int(active_count) - support_i - support_j + n11
            quarter = active_count // 4
            balance_penalty = (
                torch.abs(n00 - quarter) +
                torch.abs(n01 - quarter) +
                torch.abs(n10 - quarter) +
                torch.abs(n11 - quarter)
            )
        else:
            balance_penalty = support_per_offset[pair_i_valid] + support_per_offset[pair_j_valid]

        # Prefer larger 11 buckets when the main score ties, but keep the target/balance objective primary.
        support_bonus = int(active_count) - support_valid
        base_balance = int(balance_penalty.max().item()) + 1 if balance_penalty.numel() > 0 else 1
        base_support = int(support_bonus.max().item()) + 1 if support_bonus.numel() > 0 else 1
        score = support_gap * base_balance + balance_penalty
        score = score * base_support + support_bonus
        return score

    raise ValueError(f"Unknown sig2 score_mode {score_mode!r}")


def _run_sig2_schedule_gpu(
    residual_masks: torch.Tensor,
    residual_active: torch.Tensor,
    inputs_by_pos_and_offset: torch.Tensor,
    out_rows: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    num_sig2_rounds: int,
    use_anchor_salvage: bool,
    score_mode: str,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    bit_values: torch.Tensor,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[list[int]],
    list[int],
    int,
    int,
    int,
]:
    device = residual_masks.device
    K_vol = int(inputs_by_pos_and_offset.size(1))
    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pair_batches: list[torch.Tensor] = []
    tile_offset_batches: list[torch.Tensor] = []
    tile_group_batches: list[torch.Tensor] = []
    total_tiles = 0
    next_group_id = 0
    completed_rounds = 0

    def emit_group_positions(group_positions: torch.Tensor, group_masks: torch.Tensor) -> bool:
        nonlocal total_tiles, next_group_id
        group_popcount = ((group_masks.unsqueeze(1) & bit_values.unsqueeze(0)) != 0).sum(dim=1)
        valid = group_popcount >= 2
        if not bool(valid.any().item()):
            return False
        valid_group_positions = group_positions[valid]
        valid_group_masks = group_masks[valid]
        group_ids = torch.arange(
            next_group_id,
            next_group_id + int(valid_group_positions.size(0)),
            dtype=torch.int32,
            device=device,
        )
        tile_pairs, tile_offsets, tile_groups, tile_counts = _materialize_exact_group_tiles_gpu(
            valid_group_positions,
            valid_group_masks,
            group_ids,
            inputs_by_pos_and_offset,
            out_rows,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            bit_values=bit_values,
        )
        if tile_offsets.numel() == 0:
            return False
        tile_pair_batches.append(tile_pairs)
        tile_offset_batches.append(tile_offsets)
        tile_group_batches.append(tile_groups)
        tile_base = total_tiles
        for tile_count in tile_counts.tolist():
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            group_tile_ids = list(range(tile_base, tile_base + int(tile_count)))
            columns[target_col].extend(group_tile_ids)
            column_lengths[target_col] += int(tile_count)
            tile_base += int(tile_count)
        total_tiles += int(tile_counts.sum().item())
        next_group_id += int(valid_group_positions.size(0))
        residual_masks[valid_group_positions] &= ~valid_group_masks.unsqueeze(1)
        updated_masks = residual_masks[valid_group_positions]
        residual_active[valid_group_positions] = (
            updated_masks.unsqueeze(-1) & bit_values.view(1, 1, -1)
        ) != 0
        return True

    for _round in range(num_sig2_rounds):
        pair_i, pair_j, pair_support = _compute_pair_support_gpu(residual_active)
        if pair_support.numel() == 0:
            break
        valid_pairs = pair_support >= BM
        if not bool(valid_pairs.any().item()):
            break
        valid_idx = torch.nonzero(valid_pairs, as_tuple=False).view(-1)
        active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1)
        if active_positions.numel() == 0:
            break
        pair_i_valid = pair_i[valid_idx]
        pair_j_valid = pair_j[valid_idx]
        support_valid = pair_support[valid_idx].to(torch.int64)
        support_per_offset = residual_active.sum(dim=0, dtype=torch.int64)
        pair_key = _compute_sig2_pair_score_gpu(
            pair_i_valid,
            pair_j_valid,
            support_valid,
            support_per_offset,
            int(active_positions.numel()),
            BM,
            score_mode=score_mode,
        )
        has_pair = residual_active[active_positions][:, pair_i_valid] & residual_active[active_positions][:, pair_j_valid]
        if not bool(has_pair.any().item()):
            break
        sentinel = pair_key.max() + 1
        masked_keys = torch.where(has_pair, pair_key.unsqueeze(0), sentinel)
        best_key, best_choice = masked_keys.min(dim=1)
        has_sig = best_key < sentinel
        if not bool(has_sig.any().item()):
            break
        chosen_positions = active_positions[has_sig]
        chosen_sig = valid_idx[best_choice[has_sig]]
        popcount = residual_active[chosen_positions].sum(dim=1, dtype=torch.int64)
        mask_base = 1 << K_vol
        row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1
        refine_key = (
            ((K_vol + 1 - popcount) * mask_base) + residual_masks[chosen_positions]
        ) * row_base + out_rows[chosen_positions]
        refine_base = int(refine_key.max().item()) + 1 if refine_key.numel() > 0 else 1
        combined_key = chosen_sig.to(torch.int64) * refine_base + refine_key
        order = torch.argsort(combined_key)
        ordered_positions = chosen_positions[order]
        ordered_sig = chosen_sig[order]
        unique_sig = torch.unique_consecutive(ordered_sig)
        round_emitted = False
        for sig in unique_sig.tolist():
            bucket_positions = ordered_positions[ordered_sig == sig]
            num_groups = int(bucket_positions.numel()) // BM
            if num_groups == 0:
                continue
            group_positions = bucket_positions[: num_groups * BM].view(num_groups, BM)
            group_masks = residual_masks[group_positions[:, 0]].clone()
            for slot in range(1, BM):
                group_masks &= residual_masks[group_positions[:, slot]]
            if emit_group_positions(group_positions, group_masks):
                round_emitted = True
        if not round_emitted:
            break
        completed_rounds += 1

    if use_anchor_salvage:
        offset_support = residual_active.sum(dim=0, dtype=torch.int32)
        if bool((offset_support >= BM).any().item()):
            anchor_off = int(torch.argmax(offset_support).item())
            anchor_positions = torch.nonzero(
                residual_active[:, anchor_off], as_tuple=False).view(-1)
            if int(anchor_positions.numel()) >= BM:
                popcount = residual_active[anchor_positions].sum(dim=1, dtype=torch.int64)
                mask_base = 1 << K_vol
                row_base = int(out_rows.max().item()) + 1 if out_rows.numel() > 0 else 1
                refine_key = (
                    ((K_vol + 1 - popcount) * mask_base) + residual_masks[anchor_positions]
                ) * row_base + out_rows[anchor_positions]
                order = torch.argsort(refine_key)
                ordered_positions = anchor_positions[order]
                num_groups = int(ordered_positions.numel()) // BM
                if num_groups > 0:
                    group_positions = ordered_positions[: num_groups * BM].view(num_groups, BM)
                    group_masks = residual_masks[group_positions[:, 0]].clone()
                    for slot in range(1, BM):
                        group_masks &= residual_masks[group_positions[:, slot]]
                    valid_anchor = (group_masks & bit_values[anchor_off]) != 0
                    if bool(valid_anchor.any().item()):
                        emit_group_positions(
                            group_positions[valid_anchor],
                            group_masks[valid_anchor],
                        )

    return (
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        column_lengths,
        total_tiles,
        next_group_id,
        completed_rounds,
    )


def _build_sig2_variant_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    num_sig2_rounds: int,
    use_anchor_salvage: bool,
    score_mode: str,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    device = triplets.device
    K_vol = int(offset_counts.numel())
    total_pairs = int(offset_counts.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets.dtype, device=device)
    inputs_by_pos_and_offset[inverse, triplets[:, 2].long()] = triplets[:, 0]
    out_rows = unique_out_rows.to(dtype=torch.int64)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    residual_masks = (
        (inputs_by_pos_and_offset.ge(0).to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    )
    residual_active = inputs_by_pos_and_offset.ge(0)

    (
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        column_lengths,
        total_tiles,
        next_group_id,
        num_rounds,
    ) = _run_sig2_schedule_gpu(
        residual_masks,
        residual_active,
        inputs_by_pos_and_offset,
        out_rows,
        BM,
        num_blocks,
        num_sig2_rounds=num_sig2_rounds,
        use_anchor_salvage=use_anchor_salvage,
        score_mode=score_mode,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        bit_values=bit_values,
    )

    stage1_num_tiles = total_tiles
    tail_order = _support_sorted_mask_order_tensor(
        residual_masks,
        residual_active,
        out_rows,
    )
    for off in range(K_vol):
        leftover_positions = tail_order[residual_active[tail_order, off]]
        num_leftover = int(leftover_positions.numel())
        if num_leftover == 0:
            continue
        num_tail_tiles = (num_leftover + BM - 1) // BM
        slot_idx = torch.arange(num_leftover, device=device)
        tile_idx = slot_idx // BM
        tile_slot = slot_idx % BM
        tile_pairs_batch = torch.full(
            (num_tail_tiles, BM, 2), -1, dtype=pair_dtype, device=device)
        tile_pairs_batch[tile_idx, tile_slot, 0] = inputs_by_pos_and_offset[leftover_positions, off]
        tile_pairs_batch[tile_idx, tile_slot, 1] = out_rows[leftover_positions].to(pair_dtype)
        tile_offsets_batch = torch.full(
            (num_tail_tiles,), off, dtype=offset_dtype, device=device)
        tile_groups_batch = torch.arange(
            next_group_id,
            next_group_id + num_tail_tiles,
            dtype=torch.int32,
            device=device,
        )
        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)
        tile_base = total_tiles
        for local_tile in range(num_tail_tiles):
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_base + local_tile)
            column_lengths[target_col] += 1
        residual_masks[leftover_positions] &= ~bit_values[off]
        residual_active[leftover_positions, off] = False
        total_tiles += num_tail_tiles
        next_group_id += num_tail_tiles

    if int(residual_active.sum().item()) != 0:
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"sig2 schedule left residual pairs after tail fill: {leftover_pairs}")

    return _finalize_schedule_from_tile_batches_gpu(
        tile_pair_batches,
        tile_offset_batches,
        tile_group_batches,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        num_rounds,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        device=device,
        template_triplets=triplets,
        template_offsets=offset_counts,
    )


def _build_sig2_two_rounds_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_sig2_variant_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        num_sig2_rounds=2,
        use_anchor_salvage=False,
        score_mode="min_support",
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_sig2_three_rounds_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_sig2_variant_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        num_sig2_rounds=3,
        use_anchor_salvage=False,
        score_mode="min_support",
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_sig2_two_rounds_anchor_salvage_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_sig2_variant_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        num_sig2_rounds=2,
        use_anchor_salvage=True,
        score_mode="min_support",
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_sig2_two_rounds_target2_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_sig2_variant_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        num_sig2_rounds=2,
        use_anchor_salvage=False,
        score_mode="target_2x",
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_sig2_two_rounds_target4_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_sig2_variant_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        num_sig2_rounds=2,
        use_anchor_salvage=False,
        score_mode="target_4x",
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_sig2_two_rounds_target2_balance_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_sig2_variant_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        num_sig2_rounds=2,
        use_anchor_salvage=False,
        score_mode="target_2x_balance",
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_sig2_two_rounds_target4_balance_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_sig2_variant_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        num_sig2_rounds=2,
        use_anchor_salvage=False,
        score_mode="target_4x_balance",
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_subset_sweep_3x2_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_subset_sweep_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        subset_size_sequence=(3, 2),
        max_candidates_sequence=None,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_subset_sweep_3x2_topk_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    max_candidates_sequence: tuple[int | None, ...],
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_subset_sweep_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        subset_size_sequence=(3, 2),
        max_candidates_sequence=max_candidates_sequence,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_subset_sweep_3x2_top3only_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    max_candidates_first_pass: int,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_subset_sweep_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        subset_size_sequence=(3, 2),
        max_candidates_sequence=(max_candidates_first_pass, None),
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_subset_sweep_3only_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    return _build_subset_sweep_schedule_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        subset_size_sequence=(3,),
        max_candidates_sequence=None,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )


def _build_sig2_two_rounds_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    residual_active = inputs_by_pos_and_offset.ge(0)
    out_rows_cpu = unique_out_rows.to(dtype=torch.int64)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64))
    residual_masks = (residual_active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pairs_list: list[list[list[int]]] = []
    tile_offsets_list: list[int] = []
    tile_group_ids_list: list[int] = []
    next_group_id = 0
    num_rounds = 0

    for _round in range(2):
        pair_i, pair_j, pair_support = _compute_pair_support_cpu(residual_active)
        if pair_support.numel() == 0:
            break
        valid_pairs = pair_support >= BM
        if not bool(valid_pairs.any().item()):
            break

        support_per_offset = residual_active.sum(dim=0, dtype=torch.int32)
        sig_buckets: dict[tuple[int, int], list[int]] = {}
        active_positions = torch.nonzero(residual_masks > 0, as_tuple=False).view(-1).tolist()
        for pos in active_positions:
            best_key = None
            best_sig = None
            for idx in range(int(pair_support.numel())):
                if not bool(valid_pairs[idx].item()):
                    continue
                i = int(pair_i[idx].item())
                j = int(pair_j[idx].item())
                if not bool(residual_active[pos, i].item()) or not bool(residual_active[pos, j].item()):
                    continue
                support = int(pair_support[idx].item())
                tie = int(support_per_offset[i].item() + support_per_offset[j].item())
                key = (support, tie, i, j)
                if best_key is None or key < best_key:
                    best_key = key
                    best_sig = (i, j)
            if best_sig is not None:
                sig_buckets.setdefault(best_sig, []).append(pos)

        if not sig_buckets:
            break

        round_emitted = False
        for sig in sorted(sig_buckets):
            positions = sig_buckets[sig]
            if len(positions) < BM:
                continue
            pos_t = torch.tensor(positions, dtype=torch.long)
            popcount = residual_active[pos_t].sum(dim=1, dtype=torch.int64)
            row_base = int(out_rows_cpu.max().item()) + 1 if out_rows_cpu.numel() > 0 else 1
            mask_base = 1 << K_vol
            keys = (
                ((K_vol + 1 - popcount) * mask_base) + residual_masks[pos_t]
            ) * row_base + out_rows_cpu[pos_t]
            order = torch.argsort(keys)
            ordered_positions = pos_t[order].tolist()
            for start in range(0, len(ordered_positions), BM):
                group_positions = ordered_positions[start: start + BM]
                if len(group_positions) != BM:
                    break
                group_mask = int(residual_masks[group_positions[0]].item())
                for pos in group_positions[1:]:
                    group_mask &= int(residual_masks[pos].item())
                if group_mask == 0 or len(_mask_to_offsets(group_mask, K_vol)) <= 1:
                    continue
                next_group_id = _emit_exact_group_tiles_cpu(
                    group_positions,
                    group_mask,
                    inputs_by_pos_and_offset,
                    out_rows_cpu,
                    tile_pairs_list,
                    tile_offsets_list,
                    tile_group_ids_list,
                    columns,
                    column_lengths,
                    next_group_id,
                )
                for off in _mask_to_offsets(group_mask, K_vol):
                    residual_active[group_positions, off] = False
                    residual_masks[group_positions] -= (1 << off)
                round_emitted = True
        if not round_emitted:
            break
        num_rounds += 1

    stage1_num_tiles = len(tile_pairs_list)
    tail_order = _support_sorted_mask_order_cpu(
        residual_masks,
        residual_active,
        out_rows_cpu,
    )
    for off in range(K_vol):
        leftover_positions = [pos for pos in tail_order if bool(residual_active[pos, off].item())]
        for start in range(0, len(leftover_positions), BM):
            chunk = leftover_positions[start: start + BM]
            if not chunk:
                continue
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1
            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pos in enumerate(chunk):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "sig2 tail encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(out_rows_cpu[pos].item())]
                residual_active[pos, off] = False
                residual_masks[pos] -= (1 << off)
            tile_pairs_list.append(tile_pairs)
            target_col = min(range(num_blocks), key=lambda col: column_lengths[col])
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1

    if residual_active.any():
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"sig2 schedule left residual pairs after tail fill: {leftover_pairs}")

    return _build_schedule_from_tile_batches_cpu(
        tile_pairs_list,
        tile_offsets_list,
        tile_group_ids_list,
        columns,
        stage1_num_tiles,
        total_pairs,
        num_positions,
        BM,
        num_blocks,
        pair_dtype,
        offset_dtype,
        num_rounds,
    )


def _build_minimal_scheduled_worklist_variant_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    variant: str,
) -> dict[str, torch.Tensor | dict[str, float]]:
    if variant == "subset_first":
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "position_bundle_union":
        from .scheduled_bundle_union_experiments import (
            build_position_bundle_union_schedule_debug,
        )
        return build_position_bundle_union_schedule_debug(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "position_bundle_union_quota":
        from .scheduled_bundle_union_experiments import (
            build_position_bundle_union_quota_schedule_debug,
        )
        return build_position_bundle_union_quota_schedule_debug(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "position_bundle_union_bundle_quota":
        from .scheduled_bundle_union_experiments import (
            build_position_bundle_union_bundle_quota_schedule_debug,
        )
        return build_position_bundle_union_bundle_quota_schedule_debug(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "position_bundle_union_prune_holes":
        from .scheduled_bundle_union_experiments import (
            build_position_bundle_union_prune_holes_schedule_debug,
        )
        return build_position_bundle_union_prune_holes_schedule_debug(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "support_sorted_mask":
        if triplets.is_cuda:
            return _build_support_sorted_mask_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_support_sorted_mask_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "dplane_popcount_sort":
        if triplets.is_cuda:
            return _build_dplane_popcount_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_dplane_popcount_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "dplane_support_sorted_sort":
        if triplets.is_cuda:
            return _build_dplane_support_sorted_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_dplane_support_sorted_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "axis_marginal_support_sorted_sort":
        if triplets.is_cuda:
            return _build_axis_marginal_support_sorted_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_axis_marginal_support_sorted_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "signature_sort":
        if triplets.is_cuda:
            return _build_ordered_signature_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
                num_rounds=3,
                prefix_len=None,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_ordered_signature_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            num_rounds=3,
            prefix_len=None,
        )
    if variant == "centrality_prefix_top3":
        if triplets.is_cuda:
            return _build_ordered_signature_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
                num_rounds=3,
                prefix_len=3,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_ordered_signature_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            num_rounds=3,
            prefix_len=3,
        )
    if variant == "position_fill_popcount_all":
        if triplets.is_cuda:
            return _build_position_fill_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
                order_mode="popcount_mask",
                min_keep_support=1,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_position_fill_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            order_mode="popcount_mask",
            min_keep_support=1,
        )
    if variant == "position_fill_popcount_full_tail":
        if triplets.is_cuda:
            return _build_position_fill_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
                order_mode="popcount_mask",
                min_keep_support=BM,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_position_fill_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            order_mode="popcount_mask",
            min_keep_support=BM,
        )
    if variant == "position_fill_popcount_half_tail":
        keep_support = max((BM + 1) // 2, 1)
        if triplets.is_cuda:
            return _build_position_fill_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
                order_mode="popcount_mask",
                min_keep_support=keep_support,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_position_fill_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            order_mode="popcount_mask",
            min_keep_support=keep_support,
        )
    if variant == "position_fill_support_all":
        if triplets.is_cuda:
            return _build_position_fill_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
                order_mode="support_sorted_mask",
                min_keep_support=1,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_position_fill_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            order_mode="support_sorted_mask",
            min_keep_support=1,
        )

    def _slot_matrix_debug(
        *,
        attach_mode: str,
        keep_threshold: int,
        num_rounds: int,
        stage1_column_mode: str,
        stage1_hole_fill_mode: str = "none",
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        debug = _build_slot_matrix_schedule_cpu_debug(
            triplets.detach().cpu(),
            offset_counts.detach().cpu(),
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            attach_mode=attach_mode,
            keep_threshold=keep_threshold,
            num_rounds=num_rounds,
            stage1_column_mode=stage1_column_mode,
            stage1_hole_fill_mode=stage1_hole_fill_mode,
        )
        return _move_schedule_debug_to_device(debug, triplets.device) if triplets.is_cuda else debug

    if variant == "slot_matrix_direct":
        if triplets.is_cuda:
            debug = _build_slot_matrix_direct_cpu_debug(
                triplets.detach().cpu(),
                offset_counts.detach().cpu(),
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
            return _move_schedule_debug_to_device(debug, triplets.device)
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_slot_matrix_direct_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "slot_matrix_span_half_chunked":
        keep_threshold = max((BM + 1) // 2, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="chunked",
        )
    if variant == "slot_matrix_span_half_chunked_fill_holes":
        keep_threshold = max((BM + 1) // 2, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="chunked",
            stage1_hole_fill_mode="all",
        )
    if variant == "slot_matrix_span_q1_chunked":
        keep_threshold = max((BM + 3) // 4, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="chunked",
        )
    if variant == "slot_matrix_span_q1_chunked_restricted_refill":
        keep_threshold = max((BM + 3) // 4, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="chunked",
            stage1_hole_fill_mode="isolated",
        )
    if variant == "slot_matrix_span_q15_chunked":
        keep_threshold = max((3 * BM + 7) // 8, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="chunked",
        )
    if variant == "slot_matrix_span_q15_chunked_restricted_refill":
        keep_threshold = max((3 * BM + 7) // 8, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="chunked",
            stage1_hole_fill_mode="isolated",
        )
    if variant == "slot_matrix_span_half_r2_chunked":
        keep_threshold = max((BM + 1) // 2, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=2,
            stage1_column_mode="chunked",
        )
    if variant == "slot_matrix_span_q3_grouped_k":
        keep_threshold = max((3 * BM + 3) // 4, 1)
        return _slot_matrix_debug(
            attach_mode="phase_span",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="grouped_k",
        )
    if variant == "slot_matrix_block_half_chunked":
        keep_threshold = max((BM + 1) // 2, 1)
        return _slot_matrix_debug(
            attach_mode="phase_block",
            keep_threshold=keep_threshold,
            num_rounds=1,
            stage1_column_mode="chunked",
        )
    if variant == "slot_matrix_block_half_r2_grouped_k":
        keep_threshold = max((BM + 1) // 2, 1)
        return _slot_matrix_debug(
            attach_mode="phase_block",
            keep_threshold=keep_threshold,
            num_rounds=2,
            stage1_column_mode="grouped_k",
        )
    if variant == "sig2_two_rounds":
        if triplets.is_cuda:
            return _build_sig2_two_rounds_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_sig2_two_rounds_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "sig2_three_rounds":
        if triplets.is_cuda:
            return _build_sig2_three_rounds_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_sig2_two_rounds_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "sig2_two_rounds_anchor_salvage":
        if triplets.is_cuda:
            return _build_sig2_two_rounds_anchor_salvage_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_sig2_two_rounds_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "sig2_two_rounds_target2":
        if triplets.is_cuda:
            return _build_sig2_two_rounds_target2_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_sig2_two_rounds_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "sig2_two_rounds_target4":
        if triplets.is_cuda:
            return _build_sig2_two_rounds_target4_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_sig2_two_rounds_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "sig2_two_rounds_target2_balance":
        if triplets.is_cuda:
            return _build_sig2_two_rounds_target2_balance_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_sig2_two_rounds_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "sig2_two_rounds_target4_balance":
        if triplets.is_cuda:
            return _build_sig2_two_rounds_target4_balance_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_sig2_two_rounds_schedule_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "subset_sweep_3x2":
        if triplets.is_cuda:
            return _build_subset_sweep_3x2_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "subset_sweep_3only":
        if triplets.is_cuda:
            return _build_subset_sweep_3only_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "subset_sweep_3x2_top256":
        if triplets.is_cuda:
            return _build_subset_sweep_3x2_topk_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                max_candidates_sequence=(256, 256),
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "subset_sweep_3x2_top128":
        if triplets.is_cuda:
            return _build_subset_sweep_3x2_topk_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                max_candidates_sequence=(128, 128),
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "subset_sweep_3x2_top64":
        if triplets.is_cuda:
            return _build_subset_sweep_3x2_topk_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                max_candidates_sequence=(64, 64),
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "subset_sweep_3x2_top3_128":
        if triplets.is_cuda:
            return _build_subset_sweep_3x2_top3only_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                max_candidates_first_pass=128,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    if variant == "subset_sweep_3x2_top3_64":
        if triplets.is_cuda:
            return _build_subset_sweep_3x2_top3only_schedule_gpu_debug(
                triplets,
                offset_counts,
                BM,
                num_blocks,
                max_candidates_first_pass=64,
                pair_dtype=pair_dtype,
                offset_dtype=offset_dtype,
            )
        triplets_cpu = triplets.detach().cpu()
        offset_counts_cpu = offset_counts.detach().cpu()
        return _build_minimal_scheduled_worklist_cpu_debug(
            triplets_cpu,
            offset_counts_cpu,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    raise ValueError(f"Unknown minimal schedule variant {variant!r}")


def _build_minimal_scheduled_worklist_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """CPU reference builder with subset-first stage-1 plus tail fill."""
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    active = inputs_by_pos_and_offset.ge(0)
    out_rows = unique_out_rows.tolist()
    out_rows_tensor = unique_out_rows.to(dtype=torch.int64)
    base_sorted_positions = _sort_positions_by_active_pattern(active, out_rows)
    residual_active = active.clone()

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    tile_offsets_list: list[int] = []
    tile_pairs_list: list[list[list[int]]] = []
    tile_group_ids_list: list[int] = []
    next_group_id = 0
    num_rounds = 0

    while True:
        candidate = _pick_subset_first_candidate(
            residual_active,
            out_rows_tensor,
            BM,
            max_subset_size=3,
        )
        if candidate is None:
            break
        subset_offsets, selected_positions = candidate
        if len(selected_positions) < BM:
            break
        round_begin = len(tile_pairs_list)
        for group_start in range(0, len(selected_positions), BM):
            positions = selected_positions[group_start: group_start + BM]
            if len(positions) != BM:
                break
            group_tile_ids: list[int] = []
            group_id = next_group_id
            next_group_id += 1
            for off in subset_offsets:
                tile_id = len(tile_pairs_list)
                tile_offsets_list.append(int(off))
                tile_group_ids_list.append(group_id)
                tile_pairs = []
                for pos in positions:
                    input_row = int(inputs_by_pos_and_offset[pos, off].item())
                    if input_row < 0:
                        raise RuntimeError(
                            "intersection stage produced an invalid slot; schedule invariant broken")
                    tile_pairs.append([input_row, int(out_rows[pos])])
                    residual_active[pos, off] = False
                tile_pairs_list.append(tile_pairs)
                group_tile_ids.append(tile_id)

            if group_tile_ids:
                target_col = min(range(num_blocks), key=lambda col: len(columns[col]))
                columns[target_col].extend(group_tile_ids)

        if len(tile_pairs_list) == round_begin:
            break
        num_rounds += 1

    stage1_columns = [col.copy() for col in columns]
    stage1_flat_tile_ids = _flatten_schedule_columns_row_major(stage1_columns)

    if not stage1_flat_tile_ids and not residual_active.any():
        empty_pairs = triplets_cpu.new_empty((0, 2))
        empty_offsets = offset_counts_cpu.new_empty((0,))
        empty_groups = offset_counts_cpu.new_empty((0,))
        empty_tail = torch.empty((0,), dtype=torch.bool)
        stats = {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": float(num_positions),
            "hole_count": 0.0,
            "stage1_hole_count": 0.0,
            "stage1_fill_ratio": 0.0,
            "stage1_num_tiles": 0.0,
            "tail_num_tiles": 0.0,
            "num_rounds": 0.0,
        }
        return {
            "scheduled_pairs_cpu": empty_pairs,
            "tile_offsets_cpu": empty_offsets,
            "tile_keep_cpu": torch.empty((0,), dtype=torch.uint8),
            "stage1_pairs_cpu": empty_pairs.clone(),
            "tile_group_ids_cpu": empty_groups,
            "final_tile_is_tail_cpu": empty_tail,
            "stats": stats,
        }

    stage1_ordered_pairs = [
        [pair[:] for pair in tile_pairs_list[tile_id]]
        for tile_id in stage1_flat_tile_ids
    ]
    if stage1_ordered_pairs:
        stage1_pairs_cpu = torch.tensor(stage1_ordered_pairs, dtype=pair_dtype).view(-1, 2)
    else:
        stage1_pairs_cpu = triplets_cpu.new_empty((0, 2))
    stage1_valid_pairs = int((stage1_pairs_cpu[:, 0] >= 0).sum().item())
    stage1_hole_count = int(stage1_pairs_cpu.size(0) - stage1_valid_pairs)

    stage1_num_tiles = len(tile_pairs_list)
    for off in range(K_vol):
        leftover_positions = [
            pos for pos in base_sorted_positions
            if bool(residual_active[pos, off].item())
        ]
        for start in range(0, len(leftover_positions), BM):
            chunk = leftover_positions[start: start + BM]
            if not chunk:
                continue
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(next_group_id)
            next_group_id += 1

            tile_pairs = [[-1, -1] for _ in range(BM)]
            for slot, pos in enumerate(chunk):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row < 0:
                    raise RuntimeError(
                        "tail stage encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(out_rows[pos])]
                residual_active[pos, off] = False
            tile_pairs_list.append(tile_pairs)

            target_col = min(range(num_blocks), key=lambda col: len(columns[col]))
            columns[target_col].append(tile_id)

    if residual_active.any():
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"minimal scheduled worklist left residual pairs after tail fill: {leftover_pairs}")

    final_flat_tile_ids = _flatten_schedule_columns_row_major(columns)
    ordered_pairs = [tile_pairs_list[tile_id] for tile_id in final_flat_tile_ids]
    ordered_offsets = [tile_offsets_list[tile_id] for tile_id in final_flat_tile_ids]
    ordered_group_ids = [tile_group_ids_list[tile_id] for tile_id in final_flat_tile_ids]
    ordered_tail_flags = [tile_id >= stage1_num_tiles for tile_id in final_flat_tile_ids]
    scheduled_pairs_cpu = torch.tensor(ordered_pairs, dtype=pair_dtype).view(-1, 2)
    tile_offsets_cpu = torch.tensor(ordered_offsets, dtype=offset_dtype)
    tile_group_ids_cpu = torch.tensor(ordered_group_ids, dtype=torch.int32)
    final_tile_is_tail_cpu = torch.tensor(ordered_tail_flags, dtype=torch.bool)
    tile_keep_cpu = _compute_tile_keep_cpu(
        scheduled_pairs_cpu,
        final_tile_is_tail_cpu,
        BM,
        num_blocks,
    )

    valid_pairs = int((scheduled_pairs_cpu[:, 0] >= 0).sum().item())
    hole_count = int(scheduled_pairs_cpu.size(0) - valid_pairs)
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"minimal scheduled worklist lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    stats = {
        "num_tiles": float(tile_offsets_cpu.numel()),
        "num_blocks": float(num_blocks),
        "padding_ratio": float(scheduled_pairs_cpu.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(scheduled_pairs_cpu.size(0), 1)),
        "num_positions": float(num_positions),
        "hole_count": float(hole_count),
        "stage1_hole_count": float(stage1_hole_count),
        "stage1_fill_ratio": float(stage1_valid_pairs / max(stage1_pairs_cpu.size(0), 1)),
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(len(tile_pairs_list) - stage1_num_tiles),
        "num_rounds": float(num_rounds),
    }

    return {
        "scheduled_pairs_cpu": scheduled_pairs_cpu,
        "tile_offsets_cpu": tile_offsets_cpu,
        "tile_keep_cpu": tile_keep_cpu,
        "stage1_pairs_cpu": stage1_pairs_cpu,
        "tile_group_ids_cpu": tile_group_ids_cpu,
        "final_tile_is_tail_cpu": final_tile_is_tail_cpu,
        "stats": stats,
    }


def _build_minimal_scheduled_worklist_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """GPU-focused builder that can also return debug artifacts."""
    device = triplets.device
    K_vol = int(offset_counts.numel())
    total_pairs = int(offset_counts.sum().item())
    if total_pairs == 0:
        empty_pairs = triplets.new_empty((0, 2))
        empty_offsets = offset_counts.new_empty((0,))
        empty_groups = offset_counts.new_empty((0,), dtype=torch.int32)
        empty_tail = torch.empty((0,), dtype=torch.bool, device=device)
        stats = {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": 0.0,
            "hole_count": 0.0,
            "stage1_hole_count": 0.0,
            "stage1_fill_ratio": 0.0,
            "stage1_num_tiles": 0.0,
            "tail_num_tiles": 0.0,
            "num_rounds": 0.0,
        }
        return {
            "scheduled_pairs_cpu": empty_pairs,
            "tile_offsets_cpu": empty_offsets,
            "tile_keep_cpu": torch.empty((0,), dtype=torch.uint8, device=device),
            "stage1_pairs_cpu": empty_pairs.clone(),
            "tile_group_ids_cpu": empty_groups,
            "final_tile_is_tail_cpu": empty_tail,
            "stats": stats,
        }

    unique_out_rows, inverse = torch.unique(
        triplets[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets.dtype, device=device)
    inputs_by_pos_and_offset[inverse, triplets[:, 2].long()] = triplets[:, 0]

    active = inputs_by_pos_and_offset.ge(0)
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))
    residual_active = active.clone()
    residual_masks = (active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    residual_popcount = active.sum(dim=1, dtype=torch.int32)
    out_rows_tensor = unique_out_rows.to(dtype=torch.int64)
    base_sorted_positions = _sort_positions_by_mask_key_tensor(
        residual_masks, residual_popcount, out_rows_tensor, K_vol)
    subset_masks_bank, subset_sizes_bank = _get_subset_mask_bank(
        K_vol, 3, device)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pair_batches: list[torch.Tensor] = []
    tile_offset_batches: list[torch.Tensor] = []
    tile_group_batches: list[torch.Tensor] = []
    total_tiles = 0
    next_group_id = 0
    num_rounds = 0

    while True:
        candidate = _pick_subset_first_candidate_gpu(
            residual_active,
            residual_masks,
            residual_popcount,
            out_rows_tensor,
            bit_values,
            subset_masks_bank,
            subset_sizes_bank,
            BM,
        )
        if candidate is None:
            break
        subset_mask, selected_positions = candidate
        if selected_positions.numel() < BM:
            break

        subset_offsets = _mask_to_offsets(subset_mask, K_vol)
        subset_size = len(subset_offsets)
        if subset_size <= 1:
            break

        group_positions = selected_positions.view(-1, BM)
        num_groups = int(group_positions.size(0))
        group_out_rows = out_rows_tensor[group_positions]

        group_inputs = torch.stack(
            [inputs_by_pos_and_offset[group_positions, off] for off in subset_offsets],
            dim=1,
        )
        tile_pairs_batch = torch.stack(
            (
                group_inputs,
                group_out_rows.unsqueeze(1).expand(-1, subset_size, -1),
            ),
            dim=-1,
        ).reshape(-1, BM, 2).to(dtype=pair_dtype)
        tile_offsets_batch = torch.tensor(
            subset_offsets, dtype=offset_dtype, device=device).repeat(num_groups)
        tile_groups_batch = torch.arange(
            next_group_id,
            next_group_id + num_groups,
            dtype=torch.int32,
            device=device,
        ).repeat_interleave(subset_size)

        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)

        tile_base = total_tiles
        for group_idx in range(num_groups):
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            group_tile_ids = list(range(
                tile_base + group_idx * subset_size,
                tile_base + (group_idx + 1) * subset_size,
            ))
            columns[target_col].extend(group_tile_ids)
            column_lengths[target_col] += subset_size

        group_flat_positions = group_positions.reshape(-1)
        for off in subset_offsets:
            bit = 1 << off
            residual_active[group_positions, off] = False
            residual_masks[group_flat_positions] -= bit
            residual_popcount[group_flat_positions] -= 1

        total_tiles += num_groups * subset_size
        next_group_id += num_groups
        num_rounds += 1

    stage1_num_tiles = total_tiles

    for off in range(K_vol):
        leftover_positions = base_sorted_positions[residual_active[base_sorted_positions, off]]
        num_leftover = int(leftover_positions.numel())
        if num_leftover == 0:
            continue
        num_tail_tiles = (num_leftover + BM - 1) // BM
        slot_idx = torch.arange(num_leftover, device=device)
        tile_idx = slot_idx // BM
        tile_slot = slot_idx % BM

        tile_pairs_batch = torch.full(
            (num_tail_tiles, BM, 2), -1, dtype=pair_dtype, device=device)
        tile_pairs_batch[tile_idx, tile_slot, 0] = inputs_by_pos_and_offset[leftover_positions, off]
        tile_pairs_batch[tile_idx, tile_slot, 1] = out_rows_tensor[leftover_positions].to(pair_dtype)
        tile_offsets_batch = torch.full(
            (num_tail_tiles,), off, dtype=offset_dtype, device=device)
        tile_groups_batch = torch.arange(
            next_group_id,
            next_group_id + num_tail_tiles,
            dtype=torch.int32,
            device=device,
        )

        tile_pair_batches.append(tile_pairs_batch)
        tile_offset_batches.append(tile_offsets_batch)
        tile_group_batches.append(tile_groups_batch)

        tile_base = total_tiles
        for local_tile in range(num_tail_tiles):
            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_base + local_tile)
            column_lengths[target_col] += 1

        bit = 1 << off
        residual_active[leftover_positions, off] = False
        residual_masks[leftover_positions] -= bit
        residual_popcount[leftover_positions] -= 1
        total_tiles += num_tail_tiles
        next_group_id += num_tail_tiles

    if int(residual_popcount.sum().item()) != 0:
        leftover_pairs = int(residual_popcount.sum().item())
        raise RuntimeError(
            f"minimal scheduled worklist left residual pairs after tail fill: {leftover_pairs}")

    if total_tiles == 0:
        empty_pairs = triplets.new_empty((0, 2))
        empty_offsets = offset_counts.new_empty((0,))
        empty_groups = offset_counts.new_empty((0,), dtype=torch.int32)
        empty_tail = torch.empty((0,), dtype=torch.bool, device=device)
        stats = {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": float(num_positions),
            "hole_count": 0.0,
            "stage1_hole_count": 0.0,
            "stage1_fill_ratio": 0.0,
            "stage1_num_tiles": 0.0,
            "tail_num_tiles": 0.0,
            "num_rounds": 0.0,
        }
        return {
            "scheduled_pairs_cpu": empty_pairs,
            "tile_offsets_cpu": empty_offsets,
            "tile_keep_cpu": torch.empty((0,), dtype=torch.uint8, device=device),
            "stage1_pairs_cpu": empty_pairs.clone(),
            "tile_group_ids_cpu": empty_groups,
            "final_tile_is_tail_cpu": empty_tail,
            "stats": stats,
        }

    all_tile_pairs = torch.cat(tile_pair_batches, dim=0)
    all_tile_offsets = torch.cat(tile_offset_batches, dim=0)
    all_tile_groups = torch.cat(tile_group_batches, dim=0)
    stage1_columns = [
        [tile_id for tile_id in col if tile_id < stage1_num_tiles]
        for col in columns
    ]
    stage1_flat_tile_ids = _flatten_schedule_columns_row_major(stage1_columns)
    stage1_indices = torch.tensor(
        stage1_flat_tile_ids, dtype=torch.long, device=device)
    final_flat_tile_ids = _flatten_schedule_columns_row_major(columns)
    final_indices = torch.tensor(
        final_flat_tile_ids, dtype=torch.long, device=device)
    stage1_pairs = (
        all_tile_pairs[stage1_indices].reshape(-1, 2)
        if stage1_indices.numel() > 0
        else triplets.new_empty((0, 2))
    )
    scheduled_pairs = all_tile_pairs[final_indices].reshape(-1, 2)
    tile_offsets = all_tile_offsets[final_indices]
    tile_group_ids = all_tile_groups[final_indices]
    final_tile_is_tail = final_indices >= stage1_num_tiles
    tile_keep = _compute_tile_keep_gpu(
        scheduled_pairs,
        final_tile_is_tail,
        BM,
        num_blocks,
    )

    valid_pairs = int((scheduled_pairs[:, 0] >= 0).sum().item())
    hole_count = int(scheduled_pairs.size(0) - valid_pairs)
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"minimal scheduled worklist lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    stats = {
        "num_tiles": float(tile_offsets.numel()),
        "num_blocks": float(num_blocks),
        "padding_ratio": float(scheduled_pairs.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(scheduled_pairs.size(0), 1)),
        "num_positions": float(num_positions),
        "hole_count": float(hole_count),
        "stage1_hole_count": 0.0,
        "stage1_fill_ratio": 1.0 if stage1_num_tiles > 0 else 0.0,
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(total_tiles - stage1_num_tiles),
        "num_rounds": float(num_rounds),
    }
    return {
        "scheduled_pairs_cpu": scheduled_pairs,
        "tile_offsets_cpu": tile_offsets,
        "tile_keep_cpu": tile_keep,
        "stage1_pairs_cpu": stage1_pairs,
        "tile_group_ids_cpu": tile_group_ids,
        "final_tile_is_tail_cpu": final_tile_is_tail,
        "stats": stats,
    }


def _build_minimal_scheduled_worklist_gpu(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    debug = _build_minimal_scheduled_worklist_gpu_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )
    return (
        debug["scheduled_pairs_cpu"],
        debug["tile_offsets_cpu"],
        debug["tile_keep_cpu"],
        debug["stats"],
    )


def _build_subset_sweep_3x2_top3_64_cuda_fast(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]] | None:
    """Fast CUDA-only builder for the fixed BM=64 subset-sweep variant.

    This keeps the subset-sweep policy on device, but freezes the row ordering
    within each sweep pass to avoid the old Python chain of many tiny tensor ops.
    """
    if (
        not triplets.is_cuda
        or BM != 64
        or triplets.dtype != torch.int32
        or offset_counts.dtype != torch.int32
    ):
        return None

    scheduled_pairs, tile_offsets, tile_keep, meta = (
        _C.gtsparse3d_build_subset_sweep_3x2_top3_64_cuda(
            triplets,
            offset_counts,
            num_blocks,
        )
    )
    if meta.numel() != 4:
        raise RuntimeError("subset sweep CUDA builder returned malformed metadata")

    num_positions = int(meta[0].item())
    stage1_num_tiles = int(meta[1].item())
    total_tiles = int(meta[2].item())
    num_rounds = int(meta[3].item())
    total_pairs = int(offset_counts.sum().item())
    valid_pairs = int((scheduled_pairs[:, 0] >= 0).sum().item())
    hole_count = int(scheduled_pairs.size(0) - valid_pairs)
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"subset sweep CUDA builder lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    stats = {
        "num_tiles": float(total_tiles),
        "num_blocks": float(num_blocks),
        "padding_ratio": float(scheduled_pairs.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(scheduled_pairs.size(0), 1)),
        "num_positions": float(num_positions),
        "hole_count": float(hole_count),
        "stage1_hole_count": 0.0,
        "stage1_fill_ratio": 1.0 if stage1_num_tiles > 0 else 0.0,
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(total_tiles - stage1_num_tiles),
        "num_rounds": float(num_rounds),
    }
    return scheduled_pairs, tile_offsets, tile_keep, stats


def _build_sig2_three_rounds_cuda_fast(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]] | None:
    """Fast CUDA builder for the fixed BM=64 sig2_three_rounds variant."""
    if (
        not triplets.is_cuda
        or BM != 64
        or triplets.dtype != torch.int32
        or offset_counts.dtype != torch.int32
        or not hasattr(_C, "gtsparse3d_build_sig2_three_rounds_cuda")
    ):
        return None

    scheduled_pairs, tile_offsets, tile_keep, meta = (
        _C.gtsparse3d_build_sig2_three_rounds_cuda(
            triplets,
            offset_counts,
            num_blocks,
        )
    )
    if meta.numel() != 4:
        raise RuntimeError("sig2 CUDA builder returned malformed metadata")

    num_positions = int(meta[0].item())
    stage1_num_tiles = int(meta[1].item())
    total_tiles = int(meta[2].item())
    num_rounds = int(meta[3].item())
    total_pairs = int(offset_counts.sum().item())
    valid_pairs = int((scheduled_pairs[:, 0] >= 0).sum().item())
    hole_count = int(scheduled_pairs.size(0) - valid_pairs)
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"sig2 CUDA builder lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    stats = {
        "num_tiles": float(total_tiles),
        "num_blocks": float(num_blocks),
        "padding_ratio": float(scheduled_pairs.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(scheduled_pairs.size(0), 1)),
        "num_positions": float(num_positions),
        "hole_count": float(hole_count),
        "stage1_hole_count": 0.0,
        "stage1_fill_ratio": 1.0 if stage1_num_tiles > 0 else 0.0,
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(total_tiles - stage1_num_tiles),
        "num_rounds": float(num_rounds),
    }
    return scheduled_pairs, tile_offsets, tile_keep, stats


def build_minimal_scheduled_worklist_from_triplets(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    BM: int,
    num_blocks: int,
    schedule_variant: str = "subset_first",
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build the minimal fixed-width tile stream for the FP32 scheduled path."""
    if triplets.numel() == 0 or offset_counts.numel() == 0:
        empty_pairs = triplets.new_empty((0, 2))
        empty_offsets = offset_counts.new_empty((0,))
        empty_keep = torch.empty((0,), dtype=torch.uint8, device=triplets.device)
        return empty_pairs, empty_offsets, empty_keep, {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": 0.0,
            "hole_count": 0.0,
            "schedule_variant": schedule_variant,
        }

    if BM <= 0:
        raise ValueError("BM must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    device = triplets.device
    pair_dtype = triplets.dtype
    offset_dtype = offset_counts.dtype

    _ = offset_starts  # kept in the signature for symmetry with other builders

    total_pairs = int(offset_counts.sum().item())
    if total_pairs == 0:
        empty_pairs = triplets.new_empty((0, 2))
        empty_offsets = offset_counts.new_empty((0,))
        empty_keep = torch.empty((0,), dtype=torch.uint8, device=triplets.device)
        return empty_pairs, empty_offsets, empty_keep, {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "padding_ratio": 0.0,
            "fill_ratio": 0.0,
            "num_positions": 0.0,
            "hole_count": 0.0,
            "schedule_variant": schedule_variant,
        }

    if schedule_variant == "subset_sweep_3x2_top3_64":
        fast_result = _build_subset_sweep_3x2_top3_64_cuda_fast(
            triplets,
            offset_counts,
            BM,
            num_blocks,
        )
        if fast_result is not None:
            scheduled_pairs, tile_offsets, tile_keep, stats = fast_result
            stats = dict(stats)
            stats["schedule_variant"] = schedule_variant
            return scheduled_pairs, tile_offsets, tile_keep, stats

    if schedule_variant == "sig2_three_rounds":
        fast_result = _build_sig2_three_rounds_cuda_fast(
            triplets,
            offset_counts,
            BM,
            num_blocks,
        )
        if fast_result is not None:
            scheduled_pairs, tile_offsets, tile_keep, stats = fast_result
            stats = dict(stats)
            stats["schedule_variant"] = schedule_variant
            return scheduled_pairs, tile_offsets, tile_keep, stats

    if schedule_variant != "subset_first":
        debug = _build_minimal_scheduled_worklist_variant_debug(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            variant=schedule_variant,
        )
        stats = dict(debug["stats"])
        stats["schedule_variant"] = schedule_variant
        return (
            debug["scheduled_pairs_cpu"],
            debug["tile_offsets_cpu"],
            debug["tile_keep_cpu"],
            stats,
        )

    if triplets.is_cuda:
        return _build_minimal_scheduled_worklist_gpu(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )

    triplets_cpu = triplets.detach().cpu()
    offset_counts_cpu = offset_counts.detach().cpu()
    debug = _build_minimal_scheduled_worklist_cpu_debug(
        triplets_cpu,
        offset_counts_cpu,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )
    scheduled_pairs_cpu = debug["scheduled_pairs_cpu"]
    tile_offsets_cpu = debug["tile_offsets_cpu"]
    tile_keep_cpu = debug["tile_keep_cpu"]
    stats = debug["stats"]
    stats = dict(stats)
    stats["schedule_variant"] = schedule_variant

    return (
        scheduled_pairs_cpu.to(device=device),
        tile_offsets_cpu.to(device=device),
        tile_keep_cpu.to(device=device),
        stats,
    )


def build_minimal_scheduled_worklist(
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    config_id: int,
    reuse_mode: str = "row_selective",
    schedule_variant: str = "subset_first",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], dict[str, float]]:
    """Build the fixed-width scheduled stream from padded offset-major pairs."""
    cfg = get_scheduled_simt_config(config_id, reuse_mode=reuse_mode)
    launch_info = scheduled_expanded_worklist_launch_info(config_id, reuse_mode=reuse_mode)
    triplets, offset_starts = compact_triplet_worklist(pairs, offset_counts)
    scheduled_pairs, tile_offsets, tile_keep, stats = build_minimal_scheduled_worklist_from_triplets(
        triplets,
        offset_counts,
        offset_starts,
        BM=cfg["BM"],
        num_blocks=launch_info["grid_dim_x"],
        schedule_variant=schedule_variant,
    )
    stats = dict(stats)
    stats["schedule_variant"] = schedule_variant
    return scheduled_pairs, tile_offsets, tile_keep, launch_info, stats


def build_baseline_fixed_width_tiles_from_triplets(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    BM: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the implicit offset-major baseline as fixed-width tiles."""
    tile_pairs = []
    tile_offsets = []
    for offset in range(int(offset_counts.numel())):
        count = int(offset_counts[offset].item())
        if count == 0:
            continue
        start = int(offset_starts[offset].item())
        for entry_start in range(start, start + count, BM):
            chunk = triplets[entry_start: min(entry_start + BM, start + count), :2].tolist()
            padded = chunk + [[-1, -1]] * (BM - len(chunk))
            tile_pairs.append(padded)
            tile_offsets.append(offset)
    if not tile_pairs:
        return (
            triplets.new_empty((0, 2)),
            offset_counts.new_empty((0,)),
        )
    return (
        torch.tensor(tile_pairs, dtype=triplets.dtype).view(-1, 2),
        torch.tensor(tile_offsets, dtype=offset_counts.dtype),
    )


def evaluate_bn_outer_row_locality(
    row_tiles_pairs: torch.Tensor,
    num_blocks: int,
    BM: int,
    num_bn_tiles: int,
    tile_group_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate row-tile locality under full-row `bn`-outer traversal plus flattened tail."""
    if row_tiles_pairs.numel() == 0 or num_bn_tiles <= 0:
        return {
            "num_tiles": 0.0,
            "num_bn_tiles": float(num_bn_tiles),
            "active_blocks": 0.0,
            "same_bn_transitions": 0.0,
            "same_slot_hits": 0.0,
            "same_slot_total": 0.0,
            "same_slot_rate": 0.0,
            "any_hit_transitions": 0.0,
            "any_hit_rate": 0.0,
            "exact_tile_hits": 0.0,
            "exact_tile_rate": 0.0,
            "same_group_transitions": 0.0,
            "same_group_rate": 0.0,
        }

    row_tiles = row_tiles_pairs.view(-1, BM, 2).cpu()
    num_row_tiles = row_tiles.size(0)
    group_ids = tile_group_ids.cpu().tolist() if tile_group_ids is not None else None
    full_row_iters = num_row_tiles // num_blocks
    tail_rows = num_row_tiles % num_blocks
    tail_base = num_row_tiles - tail_rows
    tail_logical_tiles = tail_rows * num_bn_tiles

    same_bn_transitions = 0
    same_slot_hits = 0
    same_slot_total = 0
    any_hit_transitions = 0
    exact_tile_hits = 0
    same_group_transitions = 0
    active_blocks = 0

    def _accumulate_transition(prev: torch.Tensor, cur: torch.Tensor, prev_row_tile: int, cur_row_tile: int) -> None:
        nonlocal same_bn_transitions, same_slot_hits, same_slot_total
        nonlocal any_hit_transitions, exact_tile_hits, same_group_transitions
        valid = (prev[:, 0] >= 0) & (cur[:, 0] >= 0)
        valid_count = int(valid.sum().item())
        if valid_count <= 0:
            return
        hit_count = int(((prev[:, 1] == cur[:, 1]) & valid).sum().item())
        same_bn_transitions += 1
        if group_ids is not None and group_ids[prev_row_tile] == group_ids[cur_row_tile]:
            same_group_transitions += 1
        same_slot_total += valid_count
        same_slot_hits += hit_count
        if hit_count > 0:
            any_hit_transitions += 1
        if hit_count == valid_count:
            exact_tile_hits += 1

    for block in range(num_blocks):
        if full_row_iters > 0 or block < tail_logical_tiles:
            active_blocks += 1

        for _bn in range(num_bn_tiles):
            prev = None
            prev_row_tile = -1
            for row_iter in range(full_row_iters):
                row_tile = block + row_iter * num_blocks
                cur = row_tiles[row_tile]
                if prev is not None:
                    _accumulate_transition(prev, cur, prev_row_tile, row_tile)
                prev = cur
                prev_row_tile = row_tile

        seq = []
        for tail_logical in range(block, tail_logical_tiles, num_blocks):
            tail_row = tail_logical // num_bn_tiles
            bn_tile = tail_logical % num_bn_tiles
            row_tile = tail_base + tail_row
            seq.append((bn_tile, row_tile, row_tiles[row_tile]))
        for idx in range(1, len(seq)):
            prev_bn, prev_row_tile, prev = seq[idx - 1]
            curr_bn, curr_row_tile, cur = seq[idx]
            if prev_bn != curr_bn:
                continue
            _accumulate_transition(prev, cur, prev_row_tile, curr_row_tile)

    return {
        "num_tiles": float(num_row_tiles),
        "num_bn_tiles": float(num_bn_tiles),
        "active_blocks": float(active_blocks),
        "same_bn_transitions": float(same_bn_transitions),
        "same_slot_hits": float(same_slot_hits),
        "same_slot_total": float(same_slot_total),
        "same_slot_rate": float(same_slot_hits / same_slot_total) if same_slot_total > 0 else 0.0,
        "any_hit_transitions": float(any_hit_transitions),
        "any_hit_rate": float(any_hit_transitions / same_bn_transitions) if same_bn_transitions > 0 else 0.0,
        "exact_tile_hits": float(exact_tile_hits),
        "exact_tile_rate": float(exact_tile_hits / same_bn_transitions) if same_bn_transitions > 0 else 0.0,
        "same_group_transitions": float(same_group_transitions),
        "same_group_rate": float(same_group_transitions / same_bn_transitions) if same_bn_transitions > 0 else 0.0,
    }


def summarize_minimal_scheduled_locality(
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    C_out: int,
    config_id: int,
    reuse_mode: str = "row_selective",
    variant: str = "subset_first",
    scheduled_pairs: torch.Tensor | None = None,
    launch_info: dict[str, int] | None = None,
) -> dict[str, float]:
    """Compare baseline vs scheduled locality under the scheduled kernel traversal."""
    cfg = get_scheduled_simt_config(config_id, reuse_mode=reuse_mode)
    if launch_info is None:
        launch_info = scheduled_expanded_worklist_launch_info(config_id, reuse_mode=reuse_mode)

    triplets, offset_starts = compact_triplet_worklist(pairs, offset_counts)
    baseline_pairs, _baseline_offsets = build_baseline_fixed_width_tiles_from_triplets(
        triplets, offset_counts, offset_starts, BM=cfg["BM"])
    if variant == "subset_first":
        if triplets.is_cuda:
            debug = _build_minimal_scheduled_worklist_gpu_debug(
                triplets,
                offset_counts,
                cfg["BM"],
                launch_info["grid_dim_x"],
                pair_dtype=triplets.dtype,
                offset_dtype=offset_counts.dtype,
            )
        else:
            debug = _build_minimal_scheduled_worklist_cpu_debug(
                triplets.detach().cpu(),
                offset_counts.detach().cpu(),
                cfg["BM"],
                launch_info["grid_dim_x"],
                pair_dtype=triplets.dtype,
                offset_dtype=offset_counts.dtype,
            )
    else:
        debug = _build_minimal_scheduled_worklist_variant_debug(
            triplets,
            offset_counts,
            cfg["BM"],
            launch_info["grid_dim_x"],
            pair_dtype=triplets.dtype,
            offset_dtype=offset_counts.dtype,
            variant=variant,
        )
    stage1_pairs_cpu = debug["stage1_pairs_cpu"]
    final_pairs_cpu = debug["scheduled_pairs_cpu"]
    tile_group_ids_cpu = debug["tile_group_ids_cpu"]
    if scheduled_pairs is not None:
        final_pairs_cpu = scheduled_pairs.detach().cpu()

    num_bn_tiles = (C_out + cfg["BN"] - 1) // cfg["BN"]
    baseline = evaluate_bn_outer_row_locality(
        baseline_pairs, num_blocks=launch_info["grid_dim_x"], BM=cfg["BM"], num_bn_tiles=num_bn_tiles)
    stage1 = evaluate_bn_outer_row_locality(
        stage1_pairs_cpu,
        num_blocks=launch_info["grid_dim_x"],
        BM=cfg["BM"],
        num_bn_tiles=num_bn_tiles,
        tile_group_ids=tile_group_ids_cpu,
    )
    scheduled = evaluate_bn_outer_row_locality(
        final_pairs_cpu,
        num_blocks=launch_info["grid_dim_x"],
        BM=cfg["BM"],
        num_bn_tiles=num_bn_tiles,
        tile_group_ids=tile_group_ids_cpu,
    )
    build_stats = debug["stats"]

    return {
        "config_id": float(config_id),
        "BM": float(cfg["BM"]),
        "BN": float(cfg["BN"]),
        "BK": float(cfg["BK"]),
        "TM": float(cfg["TM"]),
        "TN": float(cfg["TN"]),
        "variant": variant,
        "num_blocks": float(launch_info["grid_dim_x"]),
        "num_bn_tiles": float(num_bn_tiles),
        "baseline_same_bn_transitions": baseline["same_bn_transitions"],
        "baseline_same_slot_rate": baseline["same_slot_rate"],
        "baseline_any_hit_rate": baseline["any_hit_rate"],
        "baseline_exact_tile_rate": baseline["exact_tile_rate"],
        "stage1_same_bn_transitions": stage1["same_bn_transitions"],
        "stage1_same_slot_rate": stage1["same_slot_rate"],
        "stage1_any_hit_rate": stage1["any_hit_rate"],
        "stage1_exact_tile_rate": stage1["exact_tile_rate"],
        "stage1_same_group_rate": stage1["same_group_rate"],
        "scheduled_same_bn_transitions": scheduled["same_bn_transitions"],
        "scheduled_same_slot_rate": scheduled["same_slot_rate"],
        "scheduled_any_hit_rate": scheduled["any_hit_rate"],
        "scheduled_exact_tile_rate": scheduled["exact_tile_rate"],
        "scheduled_same_group_rate": scheduled["same_group_rate"],
        "delta_stage1_same_slot_rate": stage1["same_slot_rate"] - baseline["same_slot_rate"],
        "delta_stage1_any_hit_rate": stage1["any_hit_rate"] - baseline["any_hit_rate"],
        "delta_stage1_exact_tile_rate": stage1["exact_tile_rate"] - baseline["exact_tile_rate"],
        "delta_final_vs_stage1_same_slot_rate": scheduled["same_slot_rate"] - stage1["same_slot_rate"],
        "delta_final_vs_stage1_any_hit_rate": scheduled["any_hit_rate"] - stage1["any_hit_rate"],
        "delta_final_vs_stage1_exact_tile_rate": scheduled["exact_tile_rate"] - stage1["exact_tile_rate"],
        "baseline_same_slot_hits": baseline["same_slot_hits"],
        "baseline_same_slot_total": baseline["same_slot_total"],
        "stage1_same_slot_hits": stage1["same_slot_hits"],
        "stage1_same_slot_total": stage1["same_slot_total"],
        "scheduled_same_slot_hits": scheduled["same_slot_hits"],
        "scheduled_same_slot_total": scheduled["same_slot_total"],
        "baseline_any_hit_transitions": baseline["any_hit_transitions"],
        "stage1_any_hit_transitions": stage1["any_hit_transitions"],
        "scheduled_any_hit_transitions": scheduled["any_hit_transitions"],
        "baseline_exact_tile_hits": baseline["exact_tile_hits"],
        "stage1_exact_tile_hits": stage1["exact_tile_hits"],
        "scheduled_exact_tile_hits": scheduled["exact_tile_hits"],
        "stage1_same_group_transitions": stage1["same_group_transitions"],
        "scheduled_same_group_transitions": scheduled["same_group_transitions"],
        "padding_ratio": build_stats["padding_ratio"],
        "fill_ratio": build_stats["fill_ratio"],
        "hole_count": build_stats["hole_count"],
        "stage1_hole_count": build_stats["stage1_hole_count"],
        "stage1_fill_ratio": build_stats["stage1_fill_ratio"],
        "stage1_num_tiles": build_stats["stage1_num_tiles"],
        "tail_num_tiles": build_stats["tail_num_tiles"],
        "num_rounds": build_stats["num_rounds"],
    }


def sort_triplets_within_tiles(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    BM: int,
) -> torch.Tensor:
    """Stably sort each (offset, M-tile) chunk by output_row."""
    sorted_triplets = triplets.clone()
    for offset in range(int(offset_counts.numel())):
        count = int(offset_counts[offset].item())
        if count == 0:
            continue
        start = int(offset_starts[offset].item())
        for entry_start in range(start, start + count, BM):
            entry_end = min(entry_start + BM, start + count)
            chunk = sorted_triplets[entry_start:entry_end]
            if chunk.size(0) <= 1:
                continue
            order = torch.argsort(chunk[:, 1], stable=True)
            sorted_triplets[entry_start:entry_end] = chunk[order]
    return sorted_triplets


def build_explicit_tile_schedule(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    C_out: int,
    BM: int,
    BN: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build explicit baseline tile descriptors from compact triplets.

    Returns:
        tile_descs: [num_tiles, 6] int32 with columns
            [weight_offset, entry_start, entry_count, channel_start, channel_count, signature]
        entry_schedule: [num_tiles] int32 baseline linear order
    """
    tile_rows = []
    signature_ids: dict[tuple[tuple[int, ...], int, int], int] = {}
    for offset in range(int(offset_counts.numel())):
        count = int(offset_counts[offset].item())
        if count == 0:
            continue
        start = int(offset_starts[offset].item())
        for entry_start in range(start, start + count, BM):
            entry_count = min(BM, start + count - entry_start)
            rows = triplets[entry_start: entry_start + entry_count, 1]
            rows_key = tuple(int(v) for v in rows.tolist())
            for channel_start in range(0, C_out, BN):
                channel_count = min(BN, C_out - channel_start)
                signature_key = (rows_key, channel_start, channel_count)
                if signature_key not in signature_ids:
                    signature_ids[signature_key] = len(signature_ids)
                signature = signature_ids[signature_key]
                tile_rows.append([
                    offset,
                    entry_start,
                    entry_count,
                    channel_start,
                    channel_count,
                    signature,
                ])

    if not tile_rows:
        empty = offset_counts.new_empty((0, 6))
        return empty, offset_counts.new_empty((0,))

    tile_descs = torch.tensor(tile_rows, dtype=offset_counts.dtype, device=offset_counts.device)
    entry_schedule = torch.arange(tile_descs.size(0), dtype=offset_counts.dtype, device=offset_counts.device)
    return tile_descs, entry_schedule


def evaluate_tile_schedule(
    tile_descs: torch.Tensor,
    entry_schedule: torch.Tensor,
    num_blocks: int,
) -> dict[str, float]:
    """Evaluate reuse potential and load balance for a scheduled tile stream."""
    num_tiles = int(entry_schedule.numel())
    if num_tiles == 0:
        return {
            "num_tiles": 0.0,
            "num_blocks": float(num_blocks),
            "reuse_hits": 0.0,
            "reuse_hit_rate": 0.0,
            "transitions": 0.0,
            "flush_count": 0.0,
            "writeback_reduction": 0.0,
            "load_cv": 0.0,
            "max_block_tiles": 0.0,
            "min_block_tiles": 0.0,
        }

    signatures = tile_descs[:, 5].tolist()
    block_lengths = []
    transitions = 0
    reuse_hits = 0

    for block in range(num_blocks):
        seq = [int(v) for v in entry_schedule[block::num_blocks].tolist()]
        block_lengths.append(len(seq))
        for idx in range(1, len(seq)):
            transitions += 1
            if signatures[seq[idx - 1]] == signatures[seq[idx]]:
                reuse_hits += 1

    mean_len = sum(block_lengths) / max(len(block_lengths), 1)
    if mean_len == 0:
        load_cv = 0.0
    else:
        variance = sum((length - mean_len) ** 2 for length in block_lengths) / len(block_lengths)
        load_cv = math.sqrt(variance) / mean_len

    flush_count = num_tiles - reuse_hits
    return {
        "num_tiles": float(num_tiles),
        "num_blocks": float(num_blocks),
        "reuse_hits": float(reuse_hits),
        "reuse_hit_rate": float(reuse_hits / transitions) if transitions > 0 else 0.0,
        "transitions": float(transitions),
        "flush_count": float(flush_count),
        "writeback_reduction": float(reuse_hits / num_tiles),
        "load_cv": float(load_cv),
        "max_block_tiles": float(max(block_lengths)),
        "min_block_tiles": float(min(block_lengths)),
    }


def reorder_tile_schedule_greedy(
    tile_descs: torch.Tensor,
    num_blocks: int,
    base_entry_schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reorder tiles to improve same-signature reuse under strided block traversal.

    The objective is to keep identical signatures on the same block queue while
    preserving near-perfect tile-count balance across blocks.
    """
    num_tiles = int(tile_descs.size(0))
    if num_tiles == 0:
        return tile_descs.new_empty((0,))
    if num_blocks <= 1:
        return torch.arange(num_tiles, dtype=tile_descs.dtype, device=tile_descs.device)

    if base_entry_schedule is None:
        base_entry_schedule = torch.arange(num_tiles, dtype=tile_descs.dtype, device=tile_descs.device)

    signature_to_tiles: dict[int, list[int]] = defaultdict(list)
    for tile_idx in [int(v) for v in base_entry_schedule.tolist()]:
        signature_to_tiles[int(tile_descs[tile_idx, 5].item())].append(tile_idx)

    groups = sorted(signature_to_tiles.items(), key=lambda item: (-len(item[1]), item[0]))
    base = num_tiles // num_blocks
    extra = num_tiles % num_blocks
    capacities = [base + (1 if block < extra else 0) for block in range(num_blocks)]
    block_queues: list[list[int]] = [[] for _ in range(num_blocks)]
    block_tails = [None for _ in range(num_blocks)]

    for signature, tiles in groups:
        cursor = 0
        while cursor < len(tiles):
            candidates = []
            for block in range(num_blocks):
                remaining = capacities[block] - len(block_queues[block])
                if remaining <= 0:
                    continue
                tail_match = 1 if block_tails[block] == signature else 0
                candidates.append((tail_match, remaining, -block, block))
            if not candidates:
                break
            _, remaining, _, block = max(candidates)
            take = min(remaining, len(tiles) - cursor)
            chunk = tiles[cursor: cursor + take]
            block_queues[block].extend(chunk)
            block_tails[block] = signature
            cursor += take

    max_len = max(len(queue) for queue in block_queues)
    reordered = []
    for step in range(max_len):
        for block in range(num_blocks):
            if step < len(block_queues[block]):
                reordered.append(block_queues[block][step])

    return torch.tensor(reordered, dtype=tile_descs.dtype, device=tile_descs.device)


def optimize_tile_schedule(
    tile_descs: torch.Tensor,
    num_blocks: int,
    base_entry_schedule: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Build a reuse-oriented schedule and report baseline vs optimized metrics."""
    if base_entry_schedule is None:
        base_entry_schedule = torch.arange(
            tile_descs.size(0), dtype=tile_descs.dtype, device=tile_descs.device)

    baseline_metrics = evaluate_tile_schedule(tile_descs, base_entry_schedule, num_blocks)
    optimized_schedule = reorder_tile_schedule_greedy(
        tile_descs, num_blocks, base_entry_schedule)
    optimized_metrics = evaluate_tile_schedule(tile_descs, optimized_schedule, num_blocks)
    return optimized_schedule, baseline_metrics, optimized_metrics


def build_and_optimize_tile_schedule(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    C_out: int,
    BM: int,
    BN: int,
    num_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, dict[str, float], dict[str, float]]:
    """Construct explicit tiles, then evaluate and optimize the traversal order."""
    tile_descs, baseline_schedule = build_explicit_tile_schedule(
        triplets, offset_counts, offset_starts, C_out, BM, BN)
    optimized_schedule, baseline_metrics, optimized_metrics = optimize_tile_schedule(
        tile_descs, num_blocks, baseline_schedule)
    return tile_descs, optimized_schedule, baseline_metrics, optimized_metrics


def explicit_tile_schedule_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    triplets: torch.Tensor,
    tile_descs: torch.Tensor,
    entry_schedule: torch.Tensor,
    N_out: int,
    config_id: int,
    grid_override: int = 0,
    tile_actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUDA forward for the explicit tile-schedule path."""
    if tile_actions is None:
        tile_actions = torch.zeros(entry_schedule.size(0), dtype=torch.uint8, device=features.device)
    return _C.gtsparse3d_explicit_tile_schedule_conv3d_forward(
        features, weight, bias, triplets, tile_descs, entry_schedule,
        tile_actions, N_out, config_id, grid_override)


def build_optimized_explicit_worklist_schedule(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    C_out: int,
    config_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], dict[str, float], dict[str, float]]:
    """Normalize triplets, build explicit tiles, and optimize schedule for the real launch grid."""
    cfg = get_ew_simt_config(config_id)
    launch_info = explicit_tile_schedule_launch_info(config_id)
    sorted_triplets = sort_triplets_within_tiles(
        triplets, offset_counts, offset_starts, cfg["BM"])
    tile_descs, baseline_schedule = build_explicit_tile_schedule(
        sorted_triplets, offset_counts, offset_starts, C_out, cfg["BM"], cfg["BN"])
    optimized_schedule, baseline_metrics, optimized_metrics = optimize_tile_schedule(
        tile_descs, launch_info["grid_dim_x"], baseline_schedule)
    return (
        sorted_triplets,
        tile_descs,
        optimized_schedule,
        launch_info,
        baseline_metrics,
        optimized_metrics,
    )


def optimized_explicit_worklist_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    N_out: int,
    config_id: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[str, float],
        dict[str, float],
    ]:
    """Build explicit tiles, optimize schedule, then launch the explicit CUDA path."""
    sorted_triplets, tile_descs, entry_schedule, launch_info, baseline_metrics, optimized_metrics = (
        build_optimized_explicit_worklist_schedule(
            triplets,
            offset_counts,
            offset_starts,
            C_out=weight.size(0),
            config_id=config_id,
        )
    )
    output = explicit_tile_schedule_conv3d_cuda(
        features, weight, bias, sorted_triplets, tile_descs, entry_schedule, N_out, config_id)
    return (
        output,
        sorted_triplets,
        tile_descs,
        entry_schedule,
        launch_info,
        baseline_metrics,
        optimized_metrics,
    )


def build_wave_schedule(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    C_out: int,
    BM: int,
    BN: int,
    num_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Multi-round bitmask-sort + column-fill wave scheduling.

    Round 1: group positions by BM, find intersection offsets (shared by all
    BM positions in the group), build full BM-entry tiles with position-slot
    alignment, column-fill into G columns. These tiles get 100% per-position
    reuse within each run.

    Round 2+: remaining triplets re-grouped via bitmask sort on residual
    bitmasks, column-filled again with their intersection offsets.

    Final: leftover triplets distributed evenly across columns.

    Returns:
        new_triplets, tile_descs, entry_schedule, stats
    """
    K_vol = int(offset_counts.numel())
    if K_vol == 0 or triplets.size(0) == 0:
        empty6 = triplets.new_empty((0, 6))
        empty1 = triplets.new_empty((0,))
        return triplets, empty6, empty1, {"reuse_rate": 0.0}

    device = triplets.device
    dtype = triplets.dtype
    G = num_blocks
    num_channel_tiles = (C_out + BN - 1) // BN

    # --- Build per-position bitmask + triplet index ---
    pos_bitmask: dict[int, int] = {}
    pos_tri: dict[int, dict[int, int]] = {}

    for off in range(K_vol):
        start = int(offset_starts[off].item())
        count = int(offset_counts[off].item())
        for i in range(count):
            idx = start + i
            out_idx = int(triplets[idx, 1].item())
            pos_bitmask.setdefault(out_idx, 0)
            pos_bitmask[out_idx] |= (1 << off)
            pos_tri.setdefault(out_idx, {}).setdefault(off, idx)

    consumed: set[int] = set()
    new_triplet_rows: list[list[int]] = []
    tile_rows: list[list[int]] = []
    columns: list[list[int]] = [[] for _ in range(G)]

    def consume_triplet(tri_idx: int) -> int:
        new_triplet_rows.append(triplets[tri_idx].tolist())
        consumed.add(tri_idx)
        return len(new_triplet_rows) - 1

    def emit_tile(off: int, es: int, ec: int, cs: int, cc: int) -> int:
        tile_rows.append([off, es, ec, cs, cc, 0])
        return len(tile_rows) - 1

    # --- Multi-round column fill ---
    remaining_positions = set(pos_bitmask.keys())
    round_num = 0
    max_rounds = 10

    while remaining_positions and round_num < max_rounds:
        round_num += 1

        # Residual bitmask: only unconsumed offsets
        residual: dict[int, int] = {}
        for pos in remaining_positions:
            rbm = 0
            for off, tri_idx in pos_tri.get(pos, {}).items():
                if tri_idx not in consumed:
                    rbm |= (1 << off)
            if rbm:
                residual[pos] = rbm

        if not residual:
            break

        # Bitmask sort on residual
        sorted_pos = sorted(
            residual.keys(),
            key=lambda p: (-bin(residual[p]).count('1'), residual[p]),
        )

        # Group into BM-sized position groups, find intersection offsets
        P = (len(sorted_pos) + BM - 1) // BM
        any_work = False

        for pg_idx in range(P):
            positions = sorted_pos[pg_idx * BM: (pg_idx + 1) * BM]
            if not positions:
                continue

            # Intersection = offsets shared by ALL positions in this group
            intersection = residual[positions[0]]
            for pos in positions[1:]:
                intersection &= residual[pos]
            if not intersection:
                continue

            # Extract intersection offsets
            inter_offsets: list[int] = []
            for off in range(K_vol):
                if intersection & (1 << off):
                    inter_offsets.append(off)
            if not inter_offsets:
                continue

            any_work = True

            # Build tiles: for each channel range, iterate intersection offsets.
            # Channel outer → offset inner ensures consecutive same-channel tiles
            # in the column, enabling per-position reuse across offsets.
            # Position order is fixed → slot alignment across offsets.

            # First, build M-tiles (triplet groups) per offset
            m_tiles_pg: list[tuple[int, int, int]] = []  # (off, entry_start, entry_count)
            for off in inter_offsets:
                entry_start = len(new_triplet_rows)
                for pos in positions:
                    tri_idx = pos_tri.get(pos, {}).get(off, -1)
                    if tri_idx >= 0 and tri_idx not in consumed:
                        consume_triplet(tri_idx)
                entry_count = len(new_triplet_rows) - entry_start
                if entry_count > 0:
                    m_tiles_pg.append((off, entry_start, entry_count))

            # Expand into channel tiles: ch outer, offset inner
            group_tile_ids: list[int] = []
            for ch_idx in range(num_channel_tiles):
                cs = ch_idx * BN
                cc = min(BN, C_out - cs)
                for off, es, ec in m_tiles_pg:
                    tid = emit_tile(off, es, ec, cs, cc)
                    group_tile_ids.append(tid)

            if group_tile_ids:
                col = min(range(G), key=lambda c: len(columns[c]))
                columns[col].extend(group_tile_ids)

        if not any_work:
            break

        # Update remaining
        remaining_positions = {
            pos for pos in remaining_positions
            if any(tri not in consumed for tri in pos_tri.get(pos, {}).values())
        }

    # --- Final: distribute leftover triplets evenly ---
    leftover_by_off: dict[int, list[int]] = {}
    for tri_idx in range(triplets.size(0)):
        if tri_idx not in consumed:
            off = int(triplets[tri_idx, 2].item())
            leftover_by_off.setdefault(off, []).append(tri_idx)

    for off, indices in leftover_by_off.items():
        for m_start in range(0, len(indices), BM):
            chunk = indices[m_start: m_start + BM]
            entry_start = len(new_triplet_rows)
            for tri_idx in chunk:
                consume_triplet(tri_idx)
            for ch_idx in range(num_channel_tiles):
                cs = ch_idx * BN
                cc = min(BN, C_out - cs)
                tid = emit_tile(off, entry_start, len(chunk), cs, cc)
                col = min(range(G), key=lambda c: len(columns[c]))
                columns[col].append(tid)

    # --- Build flat schedule + tile_actions via row-major interleave ---
    if not new_triplet_rows:
        empty6 = triplets.new_empty((0, 6))
        empty1 = triplets.new_empty((0,))
        empty_u8 = triplets.new_empty((0,), dtype=torch.uint8)
        return triplets, empty6, empty1, empty_u8, {"reuse_rate": 0.0}

    new_triplets = torch.tensor(new_triplet_rows, dtype=dtype, device=device)
    tile_descs = torch.tensor(tile_rows, dtype=dtype, device=device)

    # Precompute per-column tile_actions: 0=WRITE (writeback+clear after GEMM),
    # 1=KEEP (next tile reuses this acc, skip writeback).
    # A tile gets KEEP(1) if the NEXT tile in the same column has:
    #   - same channel_start and channel_count
    #   - same entry_count
    #   - identical output_row at every position slot
    # Last tile in each column is always WRITE(0).
    col_actions: list[list[int]] = []
    total_reuse = 0
    total_transitions = 0
    for col in range(G):
        actions: list[int] = []
        col_len = len(columns[col])
        for row in range(col_len):
            if row == col_len - 1:
                actions.append(0)  # last tile: always WRITE
                continue
            cur_tid = columns[col][row]
            next_tid = columns[col][row + 1]
            cur_td = tile_rows[cur_tid]
            next_td = tile_rows[next_tid]
            cur_es, cur_ec, cur_cs, cur_cc = cur_td[1], cur_td[2], cur_td[3], cur_td[4]
            next_es, next_ec, next_cs, next_cc = next_td[1], next_td[2], next_td[3], next_td[4]

            if cur_ec == 0 or next_ec == 0:
                actions.append(0)
                continue

            if cur_cs != next_cs or cur_cc != next_cc:
                actions.append(0)
                continue

            if cur_ec != next_ec:
                actions.append(0)
                total_transitions += cur_ec
                continue

            reuse = True
            for p in range(cur_ec):
                total_transitions += 1
                cur_out = int(new_triplets[cur_es + p, 1].item()) if cur_es + p < new_triplets.size(0) else -1
                next_out = int(new_triplets[next_es + p, 1].item()) if next_es + p < new_triplets.size(0) else -1
                if cur_out != next_out:
                    reuse = False
                else:
                    total_reuse += 1

            actions.append(1 if reuse else 0)
        col_actions.append(actions)

    # Interleave into flat schedule + actions
    max_rows = max((len(c) for c in columns), default=0)
    schedule_list: list[int] = []
    actions_list: list[int] = []
    for row in range(max_rows):
        for col in range(G):
            if row < len(columns[col]):
                schedule_list.append(columns[col][row])
                actions_list.append(col_actions[col][row])

    entry_schedule = torch.tensor(schedule_list, dtype=dtype, device=device)
    tile_actions = torch.tensor(actions_list, dtype=torch.uint8, device=device)

    reuse_rate = total_reuse / max(total_transitions, 1)
    stats = {
        "reuse_rate": reuse_rate,
        "total_reuse": float(total_reuse),
        "total_transitions": float(total_transitions),
        "num_tiles": float(len(tile_rows)),
        "num_schedule_entries": float(len(schedule_list)),
        "grid": float(G),
        "num_rounds": float(round_num),
    }

    return new_triplets, tile_descs, entry_schedule, tile_actions, stats


def build_wave_schedule_cuda(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    C_out: int,
    BM: int,
    BN: int,
    num_blocks: int,
    N_out: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """CUDA-accelerated wave schedule construction.

    All heavy lifting on GPU. Python side is just a thin wrapper.
    """
    K_vol = int(offset_counts.numel())
    if K_vol == 0 or triplets.size(0) == 0:
        e6 = triplets.new_empty((0, 6))
        e1 = triplets.new_empty((0,))
        eu = triplets.new_empty((0,), dtype=torch.uint8)
        return triplets, e6, e1, eu, {"reuse_rate": 0.0}

    new_triplets, tile_descs, entry_schedule, tile_actions = (
        _C.gtsparse3d_build_wave_schedule_cuda(
            triplets, offset_counts, offset_starts,
            N_out, BM, BN, C_out, num_blocks))

    # Compute reuse stats from tile_actions
    num_keep = int(tile_actions.sum().item()) if tile_actions.numel() > 0 else 0
    num_tiles = int(tile_actions.numel())
    stats = {
        "reuse_rate": num_keep / max(num_tiles, 1),
        "total_reuse": float(num_keep),
        "total_transitions": float(num_tiles),
        "num_tiles": float(tile_descs.size(0) if tile_descs.dim() == 2 else 0),
        "num_schedule_entries": float(num_tiles),
        "grid": float(num_blocks),
        "num_rounds": 1.0,
    }

    return new_triplets, tile_descs, entry_schedule, tile_actions, stats


def wave_schedule_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    offset_starts: torch.Tensor,
    N_out: int,
    config_id: int,
    use_cuda_schedule: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build wave schedule, then launch the explicit CUDA kernel.

    Returns:
        output, new_triplets, tile_descs, entry_schedule, tile_actions, stats
    """
    cfg = get_ew_simt_config(config_id)
    launch_info = explicit_tile_schedule_launch_info(config_id)
    build_fn = build_wave_schedule_cuda if use_cuda_schedule else build_wave_schedule
    if use_cuda_schedule:
        new_triplets, tile_descs, entry_schedule, tile_actions, stats = build_fn(
            triplets, offset_counts, offset_starts,
            C_out=weight.size(0),
            BM=cfg["BM"], BN=cfg["BN"],
            num_blocks=launch_info["grid_dim_x"],
            N_out=N_out,
        )
    else:
        new_triplets, tile_descs, entry_schedule, tile_actions, stats = build_fn(
            triplets, offset_counts, offset_starts,
            C_out=weight.size(0),
            BM=cfg["BM"], BN=cfg["BN"],
            num_blocks=launch_info["grid_dim_x"],
        )
    # No grid_override needed — schedule inner dim already matches
    # kernel's gridDim.x, so stride alignment is exact.
    output = explicit_tile_schedule_conv3d_cuda(
        features, weight, bias, new_triplets, tile_descs, entry_schedule,
        N_out, config_id, tile_actions=tile_actions,
    )
    return output, new_triplets, tile_descs, entry_schedule, tile_actions, stats


def build_expanded_worklist_subm(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build expanded worklist for SubM conv (padded layout).

    Returns:
        pairs: [K_vol * N, 2] int32 — padded, offset k at rows [k*N .. k*N+counts[k])
        offset_counts: [K] int32 — valid pair count per offset
    """
    coords = st.indices
    assert coords.is_contiguous(), "coords must be contiguous"
    kD, kH, kW = kernel_size

    return _C.gtsparse3d_build_expanded_worklist(
        coords,
        kD, kH, kW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2])


def build_subm_rowmap_from_coords(
    coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SubM row_inputs/row_masks directly from coords via hash lookup."""
    assert coords.is_contiguous(), "coords must be contiguous"
    kD, kH, kW = kernel_size
    return _C.gtsparse3d_build_subm_rowmap(
        coords,
        kD, kH, kW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
    )


def build_subm_rowmap_from_coords_into(
    coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build SubM row_inputs/row_masks into caller-provided tensors."""
    assert coords.is_contiguous(), "coords must be contiguous"
    kD, kH, kW = kernel_size
    return _C.gtsparse3d_build_subm_rowmap_into(
        coords,
        kD, kH, kW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        row_inputs,
        row_masks,
        offset_counts,
    )


def build_subm_rowmap(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return build_subm_rowmap_from_coords(st.indices, kernel_size, padding, dilation)


def _conv_output_spatial_shape(
    spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> list[int]:
    return [
        (int(spatial_shape[i]) + 2 * int(padding[i]) - int(dilation[i]) * (int(kernel_size[i]) - 1) - 1)
        // int(stride[i]) + 1
        for i in range(3)
    ]


def build_full_rowmap_from_coords(
    coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build full-conv row_inputs/row_masks directly from coords via output enumeration + hash lookup."""
    assert coords.is_contiguous(), "coords must be contiguous"
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    out_spatial = _conv_output_spatial_shape(spatial_shape, kernel_size, stride, padding, dilation)
    out_coords = _C.gtsparse3d_enumerate_output_coords(
        coords,
        out_spatial[0], out_spatial[1], out_spatial[2],
        kD, kH, kW,
        sD, sH, sW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
    )
    row_inputs, row_masks, offset_counts, global_offset_support = _C.gtsparse3d_build_full_rowmap(
        out_coords,
        coords,
        kD, kH, kW,
        sD, sH, sW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
    )
    return row_inputs, row_masks, offset_counts, global_offset_support, out_coords, out_spatial


def build_full_rowmap_from_coords_into(
    coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build full-conv rowmap into caller-provided tensors."""
    assert coords.is_contiguous(), "coords must be contiguous"
    kD, kH, kW = kernel_size
    sD, sH, sW = stride
    out_spatial = _conv_output_spatial_shape(spatial_shape, kernel_size, stride, padding, dilation)
    out_coords = _C.gtsparse3d_enumerate_output_coords(
        coords,
        out_spatial[0], out_spatial[1], out_spatial[2],
        kD, kH, kW,
        sD, sH, sW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
    )
    built_row_inputs, built_row_masks, built_offset_counts, global_offset_support = _C.gtsparse3d_build_full_rowmap_into(
        out_coords,
        coords,
        kD, kH, kW,
        sD, sH, sW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        row_inputs,
        row_masks,
        offset_counts,
    )
    return built_row_inputs, built_row_masks, built_offset_counts, global_offset_support, out_coords, out_spatial


def build_full_rowmap(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    return build_full_rowmap_from_coords(
        st.indices,
        st.spatial_shape,
        kernel_size,
        stride,
        padding,
        dilation,
    )


def build_inverse_rowmap_from_coords(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inverse-conv row_inputs/row_masks directly from paired coords via hash lookup."""
    assert in_coords.is_contiguous(), "in_coords must be contiguous"
    assert out_coords.is_contiguous(), "out_coords must be contiguous"
    kD, kH, kW = kernel_size
    return _C.gtsparse3d_build_inverse_rowmap(
        out_coords,
        in_coords,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
    )


def build_inverse_rowmap_from_coords_into(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build inverse-conv rowmap into caller-provided tensors."""
    assert in_coords.is_contiguous(), "in_coords must be contiguous"
    assert out_coords.is_contiguous(), "out_coords must be contiguous"
    kD, kH, kW = kernel_size
    return _C.gtsparse3d_build_inverse_rowmap_into(
        out_coords,
        in_coords,
        kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        row_inputs,
        row_masks,
        offset_counts,
    )


def build_inverse_rowmap(
    in_st: GTSparseSparseConvTensor,
    out_coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return build_inverse_rowmap_from_coords(
        in_st.indices,
        out_coords,
        kernel_size,
        stride,
        padding,
        dilation,
    )


def _rowmap_to_offset_major_pairs(
    row_inputs: torch.Tensor,
    offset_counts: torch.Tensor,
) -> torch.Tensor:
    """Materialize padded offset-major pairs from compact row_inputs."""
    if row_inputs.dtype != torch.int32 or offset_counts.dtype != torch.int32:
        raise TypeError("row_inputs and offset_counts must be int32")
    if row_inputs.dim() != 2 or offset_counts.dim() != 1:
        raise ValueError("row_inputs must be [N_out, K], offset_counts must be [K]")
    if row_inputs.size(1) != offset_counts.numel():
        raise ValueError("row_inputs.size(1) must match offset_counts.numel()")

    k_vol = int(offset_counts.numel())
    if k_vol == 0:
        return row_inputs.new_empty((0, 2))

    n_stride = int(offset_counts.max().item()) if offset_counts.numel() > 0 else 0
    if n_stride <= 0:
        return row_inputs.new_empty((0, 2))

    pairs = row_inputs.new_full((k_vol * n_stride, 2), -1)
    for off in range(k_vol):
        count = int(offset_counts[off].item())
        if count <= 0:
            continue
        rows = torch.nonzero(row_inputs[:, off] >= 0, as_tuple=False).view(-1)
        if int(rows.numel()) != count:
            raise RuntimeError(
                f"rowmap/pair count mismatch at offset {off}: rows={int(rows.numel())} counts={count}"
            )
        base = off * n_stride
        pairs[base: base + count, 0] = row_inputs[rows, off]
        pairs[base: base + count, 1] = rows.to(dtype=row_inputs.dtype)
    return pairs


def _subm_rowmap_to_offset_major_pairs(
    row_inputs: torch.Tensor,
    offset_counts: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible alias for older SubM-only callers."""
    return _rowmap_to_offset_major_pairs(row_inputs, offset_counts)


def build_expanded_triplet_worklist_subm(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build compact explicit SubM worklist entries.

    Returns:
        triplets: [nnz, 3] int32 with columns
            [input_index, output_index, weight_offset]
        offset_counts: [K] int32
        offset_starts: [K] int32
    """
    pairs, offset_counts = build_expanded_worklist_subm(
        st, kernel_size, padding, dilation)

    triplets, offset_starts = compact_triplet_worklist(pairs, offset_counts)
    return triplets, offset_counts, offset_starts


def expanded_worklist_conv3d(
    features: torch.Tensor,       # [N, C_in]
    weight: torch.Tensor,         # [C_out, C_in, kD, kH, kW] channels_last_3d
    bias: torch.Tensor | None,
    pairs: torch.Tensor,          # [K_vol * N_stride, 2] padded
    offset_counts: torch.Tensor,  # [K] int32
) -> torch.Tensor:
    """Expanded-worklist conv: per-offset gather → matmul → scatter_add (Python reference)."""
    N = features.size(0)
    C_out = weight.size(0)
    C_in = weight.size(1)
    K = offset_counts.size(0)
    N_stride = pairs.size(0) // K

    w = weight.contiguous().permute(2, 3, 4, 1, 0).reshape(K, C_in, C_out)

    output = torch.zeros(N, C_out, dtype=features.dtype, device=features.device)

    for k in range(K):
        count = offset_counts[k].item()
        if count == 0:
            continue

        p = pairs[k * N_stride : k * N_stride + count]
        input_rows = p[:, 0].long()
        output_rows = p[:, 1].long()

        gathered = features[input_rows]
        result = gathered @ w[k]
        output.scatter_add_(0, output_rows.unsqueeze(1).expand_as(result), result)

    if bias is not None:
        output += bias.unsqueeze(0)

    return output


def expanded_triplet_worklist_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    triplets: torch.Tensor,       # [nnz, 3] int32
    offset_counts: torch.Tensor,  # [K] int32
    offset_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expanded-worklist conv over compact explicit triplets."""
    N = features.size(0)
    C_out = weight.size(0)
    C_in = weight.size(1)
    K = offset_counts.size(0)

    if offset_starts is None:
        offset_starts = _offset_starts_from_counts(offset_counts)

    w = weight.contiguous().permute(2, 3, 4, 1, 0).reshape(K, C_in, C_out)
    output = torch.zeros(N, C_out, dtype=features.dtype, device=features.device)

    for k in range(K):
        count = int(offset_counts[k].item())
        if count == 0:
            continue
        start = int(offset_starts[k].item())
        entries = triplets[start: start + count]
        gathered = features[entries[:, 0].long()]
        result = gathered @ w[k]
        output.scatter_add_(0, entries[:, 1].long().unsqueeze(1).expand_as(result), result)

    if bias is not None:
        output += bias.unsqueeze(0)

    return output


def build_expanded_worklist_full(
    in_st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Build expanded worklist for full (strided) conv.

    Returns:
        pairs, offset_counts,
        out_coords: [N_out, 4] int32 — output active coordinates,
        out_spatial_shape: [D', H', W'] — output spatial shape
    """
    in_coords = in_st.indices
    assert in_coords.is_contiguous(), "in_coords must be contiguous"
    kD, kH, kW = kernel_size
    sD, sH, sW = stride

    out_spatial = _conv_output_spatial_shape(in_st.spatial_shape, kernel_size, stride, padding, dilation)

    # CUDA: enumerate output coords, worst-case alloc + unique
    out_coords = _C.gtsparse3d_enumerate_output_coords(
        in_coords,
        out_spatial[0], out_spatial[1], out_spatial[2], kD, kH, kW,
        sD, sH, sW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2])

    # Build pairs: iterate output voxels, lookup input neighbors via hash table
    # swap=false: pairs = (input_row, output_row)
    pairs, counts_t = _C.gtsparse3d_build_expanded_worklist_strided(
        out_coords, in_coords,
        kD, kH, kW,
        sD, sH, sW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        False)

    return pairs, counts_t, out_coords, out_spatial


def build_expanded_worklist_inverse(
    in_st: GTSparseSparseConvTensor,
    out_coords: torch.Tensor,
    out_spatial_shape: list,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build expanded worklist for inverse (transpose) conv.

    in_st: the inverse conv's input (= forward conv's output)
    out_coords: the inverse conv's output coords (= forward conv's input coords)
    out_spatial_shape: the inverse conv's output spatial shape

    The kernel map is the transpose of the forward conv's map.
    For offset (rd, rh, rw), input at (b, od, oh, ow) maps to
    output at (b, od*stride + rd*dilation - padding, ...).
    """
    kD, kH, kW = kernel_size
    sD, sH, sW = stride

    # Iterate input voxels, lookup output neighbors via hash table on out_coords
    # swap=true: pairs = (input_row, output_row)
    in_coords = in_st.indices
    assert in_coords.is_contiguous(), "in_coords must be contiguous"
    assert out_coords.is_contiguous(), "out_coords must be contiguous"
    pairs, counts_t = _C.gtsparse3d_build_expanded_worklist_strided(
        in_coords, out_coords,
        kD, kH, kW,
        sD, sH, sW,
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        True)

    return pairs, counts_t


def expanded_worklist_conv3d_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    active_ratio: float = -1.0,
) -> torch.Tensor:
    """Autotuned expanded-worklist Conv3d forward (CUDA kernel)."""
    C_in = features.size(1)
    C_out = weight.size(0)
    K_vol = weight.size(2) * weight.size(3) * weight.size(4)
    N_stride = pairs.size(0) // K_vol

    if active_ratio < 0:
        active_ratio = offset_counts.sum().item() / max(N_out * K_vol, 1)

    config_id = get_config_id(
        C_in, C_out, K_vol, features.dtype,
        features, weight, pairs, offset_counts, N_stride,
        N_out, active_ratio,
    )

    return _C.gtsparse3d_expanded_worklist_conv3d_forward(
        features, weight, bias, pairs, offset_counts,
        N_stride, N_out, config_id,
    )
def expanded_worklist_conv3d_scheduled_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    config_id: int = -1,
    reuse_mode: str = "row_selective",
    schedule_variant: str = "subset_first",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], dict[str, float]]:
    """Run the minimal FP32 scheduled path with full-row locality and flattened tail.

    If ``config_id < 0``, autotune chooses a SIMT config using a scheduled-path
    cache key that is specific to the requested ``reuse_mode``.
    """
    if features.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("scheduled FP32 path requires float32 features and weight")
    reuse_mode_map = {
        "off": 2,
        "exact": 1,
        "row_selective": 2,
    }
    if reuse_mode not in reuse_mode_map:
        raise ValueError("scheduled reuse_mode must be one of: off, exact, row_selective")

    if config_id < 0:
        C_in = features.size(1)
        C_out = weight.size(0)
        K_vol = weight.size(2) * weight.size(3) * weight.size(4)
        N_stride = pairs.size(0) // K_vol
        active_ratio = offset_counts.sum().item() / max(N_out * K_vol, 1)
        config_id = get_config_id(
            C_in,
            C_out,
            K_vol,
            features.dtype,
            features,
            weight,
            pairs,
            offset_counts,
            N_stride,
            N_out,
            active_ratio,
            simt_variant="scheduled",
            scheduled_reuse_mode=reuse_mode,
            scheduled_variant=schedule_variant,
        )

    scheduled_pairs, tile_offsets, tile_keep, launch_info, schedule_stats = build_minimal_scheduled_worklist(
        pairs,
        offset_counts,
        config_id=config_id,
        reuse_mode=reuse_mode,
        schedule_variant=schedule_variant,
    )
    if reuse_mode != "exact":
        tile_keep = torch.empty((0,), dtype=torch.uint8, device=scheduled_pairs.device)
    if reuse_mode == "exact":
        output = _C.gtsparse3d_expanded_worklist_conv3d_scheduled_exact_forward(
            features,
            weight,
            bias,
            scheduled_pairs,
            tile_offsets,
            tile_keep,
            N_out,
            config_id,
            launch_info["grid_dim_x"],
        )
    else:
        output = _C.gtsparse3d_expanded_worklist_conv3d_scheduled_forward(
            features,
            weight,
            bias,
            scheduled_pairs,
            tile_offsets,
            tile_keep,
            N_out,
            config_id,
            launch_info["grid_dim_x"],
            reuse_mode_map[reuse_mode],
        )
    return output, scheduled_pairs, tile_offsets, launch_info, schedule_stats


def _resolve_scheduled_bundle_config_id(
    features: torch.Tensor,
    weight: torch.Tensor,
    pairs: torch.Tensor | None,
    offset_counts: torch.Tensor,
    N_out: int,
    config_id: int,
    active_ratio: float | None,
    schedule_variant: str,
) -> int:
    if config_id >= 0:
        return int(config_id)
    if active_ratio is None:
        raise ValueError("active_ratio must be provided when config_id < 0 for scheduled bundle autotune")
    if pairs is None:
        raise ValueError("pairs must be provided when config_id < 0 for scheduled bundle autotune")
    K_vol = weight.size(2) * weight.size(3) * weight.size(4)
    N_stride = pairs.size(0) // K_vol
    return get_config_id(
        features.size(1),
        weight.size(0),
        K_vol,
        features.dtype,
        features,
        weight,
        pairs,
        offset_counts,
        N_stride,
        N_out,
        active_ratio=active_ratio,
        simt_variant="scheduled",
        scheduled_reuse_mode="row_selective",
        scheduled_variant=schedule_variant,
    )


def _make_scheduled_bundle_stats(
    scheduled_pairs: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_slot_states: torch.Tensor,
    offset_counts: torch.Tensor,
    grid_dim_x: int,
    schedule_variant: str,
    *,
    bm: int | None = None,
    bundle_tile_counts: torch.Tensor | None = None,
    total_tiles: torch.Tensor | None = None,
    selected_bundles: torch.Tensor | None = None,
    build_input_mode: str | None = None,
) -> dict[str, float]:
    total_pairs = int(offset_counts.sum().item())
    num_tiles_capacity = int(tile_offsets.numel())
    num_tiles = num_tiles_capacity
    if total_tiles is not None and total_tiles.numel() > 0:
        num_tiles = min(int(total_tiles.reshape(-1)[0].item()), num_tiles_capacity)

    resolved_bm = int(bm) if bm is not None else 0
    if resolved_bm <= 0 and num_tiles_capacity > 0:
        resolved_bm = int(scheduled_pairs.size(0) // max(num_tiles_capacity, 1))
    elif resolved_bm <= 0 and tile_slot_states.dim() == 2:
        resolved_bm = int(tile_slot_states.size(1) * 4)

    used_slots = num_tiles * resolved_bm
    used_pair_rows = min(used_slots, int(scheduled_pairs.size(0)))
    used_pairs = scheduled_pairs[:used_pair_rows]

    valid_pairs = 0
    hole_count = 0
    stage1_num_tiles = 0
    tail_num_tiles = num_tiles
    stage1_valid_pairs = 0
    stage1_hole_count = 0
    tail_valid_pairs = 0
    stage1_owner_pairs = 0
    stage1_direct_pairs = 0
    stage1_owner_slots = 0
    avg_union_size = 0.0
    max_union_size = 0.0
    selected_bundle_count = 0

    if selected_bundles is not None and selected_bundles.numel() > 0:
        selected_bundle_count = int(selected_bundles.reshape(-1)[0].item())
    elif bundle_tile_counts is not None and bundle_tile_counts.numel() > 0:
        selected_bundle_count = int(bundle_tile_counts.numel())

    if bundle_tile_counts is not None and bundle_tile_counts.numel() > 0:
        used_bundle_tile_counts = bundle_tile_counts
        if selected_bundle_count > 0 and used_bundle_tile_counts.numel() > selected_bundle_count:
            used_bundle_tile_counts = used_bundle_tile_counts[:selected_bundle_count]
        if used_bundle_tile_counts.numel() > 0:
            avg_union_size = float(used_bundle_tile_counts.float().mean().item())
            max_union_size = float(used_bundle_tile_counts.max().item())

    if num_tiles > 0 and resolved_bm > 0 and tile_slot_states.numel() > 0:
        used_tile_slot_states = tile_slot_states[:num_tiles].to(torch.int32)
        shifts = torch.tensor([0, 2, 4, 6], device=tile_slot_states.device, dtype=torch.int32)
        unpacked_states = ((used_tile_slot_states.unsqueeze(-1) >> shifts) & 0x3).reshape(num_tiles, -1)
        unpacked_states = unpacked_states[:, :resolved_bm].to(torch.uint8)

        valid_mask = unpacked_states != 3
        owner_mask = (unpacked_states == 1) | (unpacked_states == 2)
        flush_mask = unpacked_states == 2
        stage1_tile_mask = owner_mask.any(dim=1)

        valid_pairs = int(valid_mask.sum().item())
        hole_count = int(used_pair_rows - valid_pairs)
        stage1_num_tiles = int(stage1_tile_mask.sum().item())
        tail_num_tiles = num_tiles - stage1_num_tiles
        if stage1_num_tiles > 0:
            stage1_valid_pairs = int(valid_mask[stage1_tile_mask].sum().item())
            stage1_hole_count = int(stage1_num_tiles * resolved_bm - stage1_valid_pairs)
            tail_valid_pairs = valid_pairs - stage1_valid_pairs
            stage1_owner_pairs = int(owner_mask.sum().item())
            stage1_direct_pairs = int((valid_mask & ~owner_mask & stage1_tile_mask.unsqueeze(1)).sum().item())
            stage1_owner_slots = int(flush_mask.sum().item())
        else:
            tail_valid_pairs = valid_pairs
    else:
        valid_pairs = int((used_pairs[:, 0] >= 0).sum().item()) if used_pairs.numel() > 0 else 0
        hole_count = int(used_pair_rows - valid_pairs)
        tail_valid_pairs = valid_pairs

    if stage1_num_tiles == 0:
        tail_num_tiles = num_tiles
        tail_valid_pairs = valid_pairs

    stage1_owner_atomic_saved = max(stage1_owner_pairs - stage1_owner_slots, 0)
    effective_atomic_pairs = total_pairs - stage1_owner_atomic_saved
    stats = {
        "BM": float(resolved_bm),
        "num_tiles_capacity": float(num_tiles_capacity),
        "num_tiles": float(num_tiles),
        "num_blocks": float(grid_dim_x),
        "total_pairs": float(total_pairs),
        "padding_ratio": float(used_pair_rows / max(total_pairs, 1)),
        "capacity_padding_ratio": float(scheduled_pairs.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(used_pair_rows, 1)),
        "fill_ratio_lower_bound": float(total_pairs / max(scheduled_pairs.size(0), 1)),
        "capacity_sized_buffers": 1.0,
        "schedule_variant": schedule_variant,
        "hole_count": float(hole_count),
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(tail_num_tiles),
        "stage1_fill_ratio": float(stage1_valid_pairs / max(stage1_num_tiles * resolved_bm, 1)),
        "stage1_hole_count": float(stage1_hole_count),
        "stage1_owner_pairs": float(stage1_owner_pairs),
        "stage1_owner_pair_fraction": float(stage1_owner_pairs / max(total_pairs, 1)),
        "stage1_direct_pairs": float(stage1_direct_pairs),
        "stage1_direct_pair_fraction": float(stage1_direct_pairs / max(total_pairs, 1)),
        "stage1_owner_slots": float(stage1_owner_slots),
        "stage1_owner_slot_fraction": float(stage1_owner_slots / max(total_pairs, 1)),
        "stage1_owner_atomic_saved": float(stage1_owner_atomic_saved),
        "stage1_owner_atomic_saved_fraction": float(stage1_owner_atomic_saved / max(total_pairs, 1)),
        "tail_pairs": float(tail_valid_pairs),
        "tail_pair_fraction": float(tail_valid_pairs / max(total_pairs, 1)),
        "effective_atomic_pairs": float(effective_atomic_pairs),
        "effective_atomic_pair_fraction": float(effective_atomic_pairs / max(total_pairs, 1)),
        "owner_pair_per_live_slot": float(stage1_owner_pairs / max(stage1_owner_slots, 1)),
        "num_bundles": float(selected_bundle_count),
        "selected_bundles": float(selected_bundle_count),
        "avg_union_size": float(avg_union_size),
        "max_union_size": float(max_union_size),
    }
    if build_input_mode is not None:
        stats["build_input_mode"] = build_input_mode
    return stats


def _pack_scheduled_bundle_tile_headers(
    tile_offsets: torch.Tensor,
    tile_slot_states: torch.Tensor,
) -> torch.Tensor:
    if tile_offsets.dim() != 1:
        raise ValueError("tile_offsets must be [num_tiles]")
    if tile_slot_states.dim() != 2 or tile_slot_states.size(0) != tile_offsets.size(0):
        raise ValueError("tile_slot_states must be [num_tiles, ceil(BM / 4)]")
    if tile_offsets.device != tile_slot_states.device:
        raise ValueError("tile_offsets and tile_slot_states must be on the same device")

    num_tiles = int(tile_offsets.numel())
    packed_cols = int(tile_slot_states.size(1))
    packed_word_cols = (packed_cols + 3) // 4
    headers = torch.empty(
        (num_tiles, 1 + packed_word_cols),
        device=tile_offsets.device,
        dtype=torch.int32,
    )
    headers[:, 0] = tile_offsets.to(torch.int32)
    if packed_word_cols == 0:
        return headers

    packed = tile_slot_states.to(torch.int32)
    padded_cols = packed_word_cols * 4
    if padded_cols != packed_cols:
        padded = torch.full(
            (num_tiles, padded_cols),
            0xFF,
            device=tile_slot_states.device,
            dtype=torch.int32,
        )
        if packed_cols > 0:
            padded[:, :packed_cols] = packed
        packed = padded

    headers[:, 1:] = (
        packed[:, 0::4]
        | (packed[:, 1::4] << 8)
        | (packed[:, 2::4] << 16)
        | (packed[:, 3::4] << 24)
    )
    return headers


def _scheduled_bundle_tile_header_cache_key(
    tile_offsets: torch.Tensor,
    tile_slot_states: torch.Tensor,
) -> tuple[int, int, int, int]:
    return (
        id(tile_offsets),
        int(getattr(tile_offsets, "_version", 0)),
        id(tile_slot_states),
        int(getattr(tile_slot_states, "_version", 0)),
    )


def _get_scheduled_bundle_tile_headers(
    tile_offsets: torch.Tensor,
    tile_slot_states: torch.Tensor,
) -> torch.Tensor:
    key = _scheduled_bundle_tile_header_cache_key(tile_offsets, tile_slot_states)
    cached = _SCHEDULED_BUNDLE_TILE_HEADER_CACHE.get(key)
    if cached is not None:
        return cached
    headers = _pack_scheduled_bundle_tile_headers(tile_offsets, tile_slot_states)
    if len(_SCHEDULED_BUNDLE_TILE_HEADER_CACHE) >= 64:
        _SCHEDULED_BUNDLE_TILE_HEADER_CACHE.clear()
    _SCHEDULED_BUNDLE_TILE_HEADER_CACHE[key] = headers
    return headers


def build_scheduled_bundle_runtime(
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    *,
    features: torch.Tensor,
    weight: torch.Tensor,
    config_id: int = -1,
    active_ratio: float | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
    dict[str, float] | None,
]:
    """Build bundle-owner schedule/runtime metadata from offset-major pairs.

    If ``config_id < 0``, callers must pass ``active_ratio`` explicitly.
    """
    if features.dtype != weight.dtype or features.dtype not in (torch.float32, torch.float16):
        raise TypeError("scheduled bundle-owner path requires matching float32 or float16 features and weight")
    if schedule_variant != "position_bundle_union_prune_holes":
        raise ValueError("scheduled bundle-owner path currently supports only position_bundle_union_prune_holes")

    from .scheduled_bundle_union_experiments import (
        _build_position_bundle_union_prune_holes_owner_cuda_runtime_fast,
    )

    resolved_config_id = _resolve_scheduled_bundle_config_id(
        features,
        weight,
        pairs,
        offset_counts,
        N_out,
        config_id,
        active_ratio,
        schedule_variant,
    )
    cfg = get_scheduled_bundle_config(resolved_config_id, dtype=features.dtype, reuse_mode="row_selective")
    launch_info = scheduled_bundle_launch_info(
        resolved_config_id,
        dtype=features.dtype,
        reuse_mode="row_selective",
    )
    if num_blocks is not None:
        launch_info = dict(launch_info)
        launch_info["grid_dim_x"] = int(num_blocks)

    runtime = _build_position_bundle_union_prune_holes_owner_cuda_runtime_fast(
        pairs,
        offset_counts,
        cfg["BM"],
        launch_info["grid_dim_x"],
        N_out,
        pair_dtype=pairs.dtype,
        offset_dtype=offset_counts.dtype,
    )
    if runtime is None:
        raise RuntimeError("scheduled bundle runtime builder from pairs is unavailable")

    scheduled_pairs = runtime["scheduled_pairs"]
    tile_offsets = runtime["tile_offsets"]
    tile_slot_states = runtime["tile_slot_states"]
    bundle_tile_counts = runtime["bundle_tile_counts"]
    total_tiles = runtime["total_tiles"]
    selected_bundles = runtime["selected_bundles"]
    _get_scheduled_bundle_tile_headers(tile_offsets, tile_slot_states)
    schedule_stats = (
        _make_scheduled_bundle_stats(
            scheduled_pairs,
            tile_offsets,
            tile_slot_states,
            offset_counts,
            launch_info["grid_dim_x"],
            schedule_variant,
            bm=cfg["BM"],
            bundle_tile_counts=bundle_tile_counts,
            total_tiles=total_tiles,
            selected_bundles=selected_bundles,
        )
        if collect_schedule_stats else None
    )
    return (
        int(resolved_config_id),
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        launch_info,
        schedule_stats,
    )


def build_scheduled_bundle_runtime_from_rows(
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    *,
    features: torch.Tensor,
    weight: torch.Tensor,
    config_id: int = -1,
    active_ratio: float | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    global_offset_support: torch.Tensor | None = None,
    collect_schedule_stats: bool = True,
    build_input_mode: str = "subm_rowmap_direct",
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
    dict[str, float] | None,
]:
    """Build bundle-owner schedule/runtime metadata from compact rowmap inputs.

    If ``config_id < 0``, callers must pass ``active_ratio`` explicitly.
    """
    if features.dtype != weight.dtype or features.dtype not in (torch.float32, torch.float16):
        raise TypeError("scheduled bundle-owner path requires matching float32 or float16 features and weight")
    if schedule_variant != "position_bundle_union_prune_holes":
        raise ValueError("scheduled bundle-owner path currently supports only position_bundle_union_prune_holes")

    autotune_pairs: torch.Tensor | None = None
    if config_id < 0:
        autotune_pairs = _rowmap_to_offset_major_pairs(row_inputs, offset_counts)
    resolved_config_id = _resolve_scheduled_bundle_config_id(
        features,
        weight,
        autotune_pairs,
        offset_counts,
        N_out,
        config_id,
        active_ratio,
        schedule_variant,
    )

    from .scheduled_bundle_union_experiments import (
        _build_position_bundle_union_prune_holes_owner_from_rows_cuda_runtime_fast,
    )

    cfg = get_scheduled_bundle_config(resolved_config_id, dtype=features.dtype, reuse_mode="row_selective")
    launch_info = scheduled_bundle_launch_info(
        resolved_config_id,
        dtype=features.dtype,
        reuse_mode="row_selective",
    )
    if num_blocks is not None:
        launch_info = dict(launch_info)
        launch_info["grid_dim_x"] = int(num_blocks)

    runtime = _build_position_bundle_union_prune_holes_owner_from_rows_cuda_runtime_fast(
        row_inputs,
        row_masks,
        offset_counts,
        cfg["BM"],
        launch_info["grid_dim_x"],
        global_offset_support=global_offset_support,
        pair_dtype=row_inputs.dtype,
        offset_dtype=offset_counts.dtype,
    )
    if runtime is None:
        raise RuntimeError("scheduled bundle runtime builder from rowmap is unavailable")

    scheduled_pairs = runtime["scheduled_pairs"]
    tile_offsets = runtime["tile_offsets"]
    tile_slot_states = runtime["tile_slot_states"]
    bundle_tile_counts = runtime["bundle_tile_counts"]
    total_tiles = runtime["total_tiles"]
    selected_bundles = runtime["selected_bundles"]
    _get_scheduled_bundle_tile_headers(tile_offsets, tile_slot_states)
    schedule_stats = (
        _make_scheduled_bundle_stats(
            scheduled_pairs,
            tile_offsets,
            tile_slot_states,
            offset_counts,
            launch_info["grid_dim_x"],
            schedule_variant,
            bm=cfg["BM"],
            bundle_tile_counts=bundle_tile_counts,
            total_tiles=total_tiles,
            selected_bundles=selected_bundles,
            build_input_mode=build_input_mode,
        )
        if collect_schedule_stats else None
    )
    return (
        int(resolved_config_id),
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        launch_info,
        schedule_stats,
    )


def _active_ratio_from_rowmap(offset_counts: torch.Tensor, n_out: int) -> float:
    k_vol = int(offset_counts.numel())
    if n_out <= 0 or k_vol <= 0:
        return 0.0
    return float(int(offset_counts.sum().item())) / float(max(n_out * k_vol, 1))


def build_scheduled_bundle_runtime_full_from_coords(
    coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    features: torch.Tensor,
    weight: torch.Tensor,
    config_id: int = -1,
    active_ratio: float | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
    dict[str, float] | None,
    torch.Tensor,
    list[int],
]:
    """Build full-conv scheduled bundle runtime directly from input coords."""
    (
        row_inputs,
        row_masks,
        offset_counts,
        global_offset_support,
        out_coords,
        out_spatial,
    ) = build_full_rowmap_from_coords(
        coords,
        spatial_shape,
        kernel_size,
        stride,
        padding,
        dilation,
    )
    n_out = int(out_coords.size(0))
    if config_id < 0 and active_ratio is None:
        active_ratio = _active_ratio_from_rowmap(offset_counts, n_out)
    runtime = build_scheduled_bundle_runtime_from_rows(
        row_inputs,
        row_masks,
        offset_counts,
        n_out,
        features=features,
        weight=weight,
        config_id=config_id,
        active_ratio=active_ratio,
        schedule_variant=schedule_variant,
        num_blocks=num_blocks,
        global_offset_support=global_offset_support,
        collect_schedule_stats=collect_schedule_stats,
        build_input_mode="full_rowmap_direct",
    )
    return (*runtime, out_coords, out_spatial)


def build_scheduled_bundle_runtime_inverse_from_coords(
    in_coords: torch.Tensor,
    out_coords: torch.Tensor,
    out_spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    *,
    features: torch.Tensor,
    weight: torch.Tensor,
    config_id: int = -1,
    active_ratio: float | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> tuple[
    int,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, int],
    dict[str, float] | None,
]:
    """Build inverse-conv scheduled bundle runtime directly from paired coords."""
    row_inputs, row_masks, offset_counts, global_offset_support = build_inverse_rowmap_from_coords(
        in_coords,
        out_coords,
        kernel_size,
        stride,
        padding,
        dilation,
    )
    n_out = int(out_coords.size(0))
    if config_id < 0 and active_ratio is None:
        active_ratio = _active_ratio_from_rowmap(offset_counts, n_out)
    return build_scheduled_bundle_runtime_from_rows(
        row_inputs,
        row_masks,
        offset_counts,
        n_out,
        features=features,
        weight=weight,
        config_id=config_id,
        active_ratio=active_ratio,
        schedule_variant=schedule_variant,
        num_blocks=num_blocks,
        global_offset_support=global_offset_support,
        collect_schedule_stats=collect_schedule_stats,
        build_input_mode="inverse_rowmap_direct",
    )


def scheduled_bundle_conv3d_forward_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scheduled_pairs: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_slot_states: torch.Tensor,
    total_tiles: torch.Tensor,
    N_out: int,
    config_id: int,
    grid_dim_x: int,
) -> torch.Tensor:
    """Launch the scheduled bundle-owner kernel with prebuilt runtime metadata."""
    if features.dtype != weight.dtype:
        raise TypeError("scheduled bundle forward requires matching feature/weight dtype")
    tile_headers = _get_scheduled_bundle_tile_headers(tile_offsets, tile_slot_states)
    if features.dtype == torch.float32:
        return _C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_forward(
            features,
            weight,
            bias,
            scheduled_pairs,
            tile_offsets,
            tile_slot_states,
            total_tiles,
            N_out,
            config_id,
            grid_dim_x,
            tile_headers,
        )
    if features.dtype == torch.float16:
        return _C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_fp16_forward(
            features,
            weight,
            bias,
            scheduled_pairs,
            tile_offsets,
            tile_slot_states,
            total_tiles,
            N_out,
            config_id,
            grid_dim_x,
            tile_headers,
        )
    raise TypeError("scheduled bundle forward only supports float32 or float16")


def scheduled_bundle_conv3d_forward_dev_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scheduled_pairs: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_slot_states: torch.Tensor,
    total_tiles: torch.Tensor,
    N_out: int,
    config_id: int,
    grid_dim_x: int,
) -> torch.Tensor:
    """Launch the dev scheduled bundle-owner kernel with prebuilt runtime metadata."""
    if features.dtype != weight.dtype:
        raise TypeError("scheduled bundle dev forward requires matching feature/weight dtype")
    if features.dtype != torch.float32:
        raise TypeError("scheduled bundle dev forward only supports float32")
    tile_headers = _get_scheduled_bundle_tile_headers(tile_offsets, tile_slot_states)
    return _C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_forward_dev(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        N_out,
        config_id,
        grid_dim_x,
        tile_headers,
    )


def scheduled_bundle_conv3d_forward_dev2_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    scheduled_pairs: torch.Tensor,
    tile_offsets: torch.Tensor,
    tile_slot_states: torch.Tensor,
    total_tiles: torch.Tensor,
    N_out: int,
    config_id: int,
    grid_dim_x: int,
) -> torch.Tensor:
    """Launch the dev2 scheduled bundle-owner kernel with prebuilt runtime metadata."""
    if features.dtype != weight.dtype:
        raise TypeError("scheduled bundle dev2 forward requires matching feature/weight dtype")
    if features.dtype != torch.float32:
        raise TypeError("scheduled bundle dev2 forward only supports float32")
    tile_headers = _get_scheduled_bundle_tile_headers(tile_offsets, tile_slot_states)
    return _C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_forward_dev2(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        N_out,
        config_id,
        grid_dim_x,
        tile_headers,
    )


def expanded_worklist_conv3d_scheduled_bundle_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    config_id: int = -1,
    *,
    active_ratio: float | None = None,
    bundle_runtime: tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[str, float],
    ] | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, float] | None,
    ]:
    """Convenience wrapper: build schedule/runtime if needed, then launch kernel.

    If ``config_id < 0``, callers must pass ``active_ratio`` explicitly.
    """
    if bundle_runtime is None:
        bundle_runtime = build_scheduled_bundle_runtime(
            pairs,
            offset_counts,
            N_out,
            features=features,
            weight=weight,
            config_id=config_id,
            active_ratio=active_ratio,
            schedule_variant=schedule_variant,
            num_blocks=num_blocks,
            collect_schedule_stats=collect_schedule_stats,
        )
    (
        resolved_config_id,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        launch_info,
        schedule_stats,
    ) = bundle_runtime
    output = scheduled_bundle_conv3d_forward_cuda(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        N_out,
        resolved_config_id,
        launch_info["grid_dim_x"],
    )
    runtime_meta = {
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }
    return (
        output,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        runtime_meta,
        launch_info,
        schedule_stats,
    )


def expanded_worklist_conv3d_scheduled_bundle_dev_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    config_id: int = -1,
    *,
    active_ratio: float | None = None,
    bundle_runtime: tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[str, float],
    ] | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, float] | None,
    ]:
    """Convenience wrapper: build runtime via mainline scheduler, launch dev FP32 kernel."""
    if features.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("scheduled bundle dev wrapper currently supports float32 only")
    if bundle_runtime is None:
        bundle_runtime = build_scheduled_bundle_runtime(
            pairs,
            offset_counts,
            N_out,
            features=features,
            weight=weight,
            config_id=config_id,
            active_ratio=active_ratio,
            schedule_variant=schedule_variant,
            num_blocks=num_blocks,
            collect_schedule_stats=collect_schedule_stats,
        )
    (
        resolved_config_id,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        _launch_info,
        schedule_stats,
    ) = bundle_runtime
    launch_info = scheduled_expanded_worklist_dev_launch_info(
        resolved_config_id, reuse_mode="row_selective")
    output = scheduled_bundle_conv3d_forward_dev_cuda(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        N_out,
        resolved_config_id,
        launch_info["grid_dim_x"],
    )
    runtime_meta = {
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }
    return (
        output,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        runtime_meta,
        launch_info,
        schedule_stats,
    )


def expanded_worklist_conv3d_scheduled_bundle_dev2_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    config_id: int = -1,
    *,
    active_ratio: float | None = None,
    bundle_runtime: tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[str, float],
    ] | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, float] | None,
    ]:
    """Convenience wrapper: build runtime via mainline scheduler, launch dev2 FP32 kernel."""
    if features.dtype != torch.float32 or weight.dtype != torch.float32:
        raise TypeError("scheduled bundle dev2 wrapper currently supports float32 only")
    if bundle_runtime is None:
        bundle_runtime = build_scheduled_bundle_runtime(
            pairs,
            offset_counts,
            N_out,
            features=features,
            weight=weight,
            config_id=config_id,
            active_ratio=active_ratio,
            schedule_variant=schedule_variant,
            num_blocks=num_blocks,
            collect_schedule_stats=collect_schedule_stats,
        )
    (
        resolved_config_id,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        _launch_info,
        schedule_stats,
    ) = bundle_runtime
    launch_info = scheduled_expanded_worklist_dev2_launch_info(
        resolved_config_id, reuse_mode="row_selective")
    output = scheduled_bundle_conv3d_forward_dev2_cuda(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        N_out,
        resolved_config_id,
        launch_info["grid_dim_x"],
    )
    runtime_meta = {
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }
    return (
        output,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        runtime_meta,
        launch_info,
        schedule_stats,
    )


def expanded_worklist_conv3d_scheduled_bundle_subm_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    coords: torch.Tensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    config_id: int = -1,
    *,
    active_ratio: float | None = None,
    bundle_runtime: tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[str, float],
    ] | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, float] | None,
]:
    """SubM scheduled bundle path using direct coords -> rowmap preprocessing.

    If ``config_id < 0``, callers must pass ``active_ratio`` explicitly.
    """
    if features.dtype != weight.dtype or features.dtype not in (torch.float32, torch.float16):
        raise TypeError("scheduled bundle-owner path requires matching float32 or float16 features and weight")
    if schedule_variant != "position_bundle_union_prune_holes":
        raise ValueError("scheduled bundle-owner path currently supports only position_bundle_union_prune_holes")
    if coords.dtype != torch.int32 or coords.dim() != 2 or coords.size(1) != 4:
        raise TypeError("coords must be [N, 4] int32")
    if not coords.is_cuda or coords.device != features.device:
        raise ValueError("coords must be a CUDA tensor on the same device as features")
    if coords.size(0) != features.size(0):
        raise ValueError("coords.size(0) must match features.size(0) for SubM scheduled bundle path")

    N_out = int(coords.size(0))
    if bundle_runtime is None:
        row_inputs, row_masks, offset_counts, global_offset_support = build_subm_rowmap_from_coords(
            coords,
            kernel_size,
            padding,
            dilation,
        )
        bundle_runtime = build_scheduled_bundle_runtime_from_rows(
            row_inputs,
            row_masks,
            offset_counts,
            N_out,
            features=features,
            weight=weight,
            config_id=config_id,
            active_ratio=active_ratio,
            schedule_variant=schedule_variant,
            num_blocks=num_blocks,
            global_offset_support=global_offset_support,
            collect_schedule_stats=collect_schedule_stats,
        )

    (
        resolved_config_id,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        launch_info,
        schedule_stats,
    ) = bundle_runtime
    output = scheduled_bundle_conv3d_forward_cuda(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        N_out,
        resolved_config_id,
        launch_info["grid_dim_x"],
    )
    runtime_meta = {
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }
    return (
        output,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        runtime_meta,
        launch_info,
        schedule_stats,
    )


def expanded_worklist_conv3d_scheduled_bundle_full_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    coords: torch.Tensor,
    spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    config_id: int = -1,
    *,
    active_ratio: float | None = None,
    bundle_runtime: tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[str, float] | None,
        torch.Tensor,
        list[int],
    ] | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, float] | None,
]:
    """Full-conv scheduled bundle path using direct coords -> out_coords/rowmap preprocessing."""
    if features.dtype != weight.dtype or features.dtype not in (torch.float32, torch.float16):
        raise TypeError("scheduled bundle-owner path requires matching float32 or float16 features and weight")
    if schedule_variant != "position_bundle_union_prune_holes":
        raise ValueError("scheduled bundle-owner path currently supports only position_bundle_union_prune_holes")
    if coords.dtype != torch.int32 or coords.dim() != 2 or coords.size(1) != 4:
        raise TypeError("coords must be [N, 4] int32")
    if not coords.is_cuda or coords.device != features.device:
        raise ValueError("coords must be a CUDA tensor on the same device as features")
    if coords.size(0) != features.size(0):
        raise ValueError("coords.size(0) must match features.size(0) for full scheduled bundle path")

    if bundle_runtime is None:
        bundle_runtime = build_scheduled_bundle_runtime_full_from_coords(
            coords,
            spatial_shape,
            kernel_size,
            stride,
            padding,
            dilation,
            features=features,
            weight=weight,
            config_id=config_id,
            active_ratio=active_ratio,
            schedule_variant=schedule_variant,
            num_blocks=num_blocks,
            collect_schedule_stats=collect_schedule_stats,
        )

    (
        resolved_config_id,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        launch_info,
        schedule_stats,
        out_coords,
        out_spatial,
    ) = bundle_runtime
    n_out = int(out_coords.size(0))
    output = scheduled_bundle_conv3d_forward_cuda(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        n_out,
        resolved_config_id,
        launch_info["grid_dim_x"],
    )
    runtime_meta = {
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }
    return (
        output,
        out_coords,
        out_spatial,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        runtime_meta,
        launch_info,
        schedule_stats,
    )


def expanded_worklist_conv3d_scheduled_bundle_inverse_cuda(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    coords: torch.Tensor,
    out_coords: torch.Tensor,
    out_spatial_shape: Tuple[int, int, int] | list[int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    config_id: int = -1,
    *,
    active_ratio: float | None = None,
    bundle_runtime: tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, int],
        dict[str, float] | None,
    ] | None = None,
    schedule_variant: str = "position_bundle_union_prune_holes",
    num_blocks: int | None = None,
    collect_schedule_stats: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    list[int],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, int],
    dict[str, float] | None,
]:
    """Inverse-conv scheduled bundle path using paired coords from a forward conv."""
    if features.dtype != weight.dtype or features.dtype not in (torch.float32, torch.float16):
        raise TypeError("scheduled bundle-owner path requires matching float32 or float16 features and weight")
    if schedule_variant != "position_bundle_union_prune_holes":
        raise ValueError("scheduled bundle-owner path currently supports only position_bundle_union_prune_holes")
    if coords.dtype != torch.int32 or coords.dim() != 2 or coords.size(1) != 4:
        raise TypeError("coords must be [N, 4] int32")
    if out_coords.dtype != torch.int32 or out_coords.dim() != 2 or out_coords.size(1) != 4:
        raise TypeError("out_coords must be [N_out, 4] int32")
    if not coords.is_cuda or coords.device != features.device or out_coords.device != features.device:
        raise ValueError("coords and out_coords must be CUDA tensors on the same device as features")
    if coords.size(0) != features.size(0):
        raise ValueError("coords.size(0) must match features.size(0) for inverse scheduled bundle path")

    if bundle_runtime is None:
        bundle_runtime = build_scheduled_bundle_runtime_inverse_from_coords(
            coords,
            out_coords,
            out_spatial_shape,
            kernel_size,
            stride,
            padding,
            dilation,
            features=features,
            weight=weight,
            config_id=config_id,
            active_ratio=active_ratio,
            schedule_variant=schedule_variant,
            num_blocks=num_blocks,
            collect_schedule_stats=collect_schedule_stats,
        )

    (
        resolved_config_id,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        selected_bundles,
        launch_info,
        schedule_stats,
    ) = bundle_runtime
    n_out = int(out_coords.size(0))
    output = scheduled_bundle_conv3d_forward_cuda(
        features,
        weight,
        bias,
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        total_tiles,
        n_out,
        resolved_config_id,
        launch_info["grid_dim_x"],
    )
    runtime_meta = {
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }
    return (
        output,
        out_coords,
        list(out_spatial_shape),
        scheduled_pairs,
        tile_offsets,
        tile_slot_states,
        runtime_meta,
        launch_info,
        schedule_stats,
    )
