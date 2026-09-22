from __future__ import annotations

from collections import deque

import torch

from gtsparse import _C

from .expanded_worklist import (
    _build_minimal_scheduled_worklist_cpu_debug,
    _build_schedule_from_tile_batches_cpu,
    _finalize_schedule_from_tile_batches_gpu,
    _move_schedule_debug_to_device,
    _ordered_offsets_by_group_support_cpu,
    _sort_positions_by_active_pattern,
    _sort_positions_by_mask_key_tensor,
    build_baseline_fixed_width_tiles_from_triplets,
    compact_triplet_worklist,
    evaluate_bn_outer_row_locality,
    get_scheduled_simt_config,
    scheduled_expanded_worklist_launch_info,
)


def _build_position_bundle_union_prune_holes_owner_cuda_fast(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor | dict[str, float]] | None:
    """Fast custom-CUDA builder for the prune_holes owner-bundle path."""
    if (
        not triplets.is_cuda
        or triplets.dtype != torch.int32
        or offset_counts.dtype != torch.int32
        or not hasattr(_C, "gtsparse3d_build_bundle_union_prune_holes_owner_cuda")
    ):
        return None

    scheduled_pairs, tile_offsets, tile_bundle_ids, bundle_owner_rows, final_tile_is_tail, meta = (
        _C.gtsparse3d_build_bundle_union_prune_holes_owner_cuda(
            triplets,
            offset_counts,
            BM,
            num_blocks,
        )
    )
    if meta.numel() != 10:
        raise RuntimeError("bundle-union CUDA builder returned malformed metadata")

    total_pairs = int(offset_counts.sum().item())
    valid_pairs = int((scheduled_pairs[:, 0] >= 0).sum().item())
    if valid_pairs != total_pairs:
        raise RuntimeError(
            f"bundle-union CUDA builder lost pairs: scheduled={valid_pairs} expected={total_pairs}")

    num_positions = int(meta[0].item())
    stage1_num_tiles = int(meta[1].item())
    total_tiles = int(meta[2].item())
    stage1_owner_pairs = int(meta[3].item())
    stage1_owner_slots = int(meta[4].item())
    stage1_refilled_pairs = int(meta[5].item())
    stage1_holes_pre_refill = int(meta[6].item())
    candidate_stage1_tiles = int(meta[7].item())
    selected_bundles = int(meta[8].item())
    num_full_bundles = int(meta[9].item())

    row_tiles = scheduled_pairs.view(-1, BM, 2)
    stage1_pairs = row_tiles[~final_tile_is_tail].reshape(-1, 2)
    stage1_valid_pairs = int((stage1_pairs[:, 0] >= 0).sum().item()) if stage1_pairs.numel() > 0 else 0
    stage1_hole_count = int(stage1_pairs.size(0) - stage1_valid_pairs)
    hole_count = int(scheduled_pairs.size(0) - valid_pairs)

    union_sizes = torch.empty((0,), dtype=torch.int64, device=triplets.device)
    if selected_bundles > 0 and tile_bundle_ids.numel() > 0:
        stage1_bundle_ids = tile_bundle_ids[tile_bundle_ids >= 0].to(torch.int64)
        if stage1_bundle_ids.numel() > 0:
            union_sizes = torch.bincount(stage1_bundle_ids, minlength=selected_bundles)

    stats = {
        "num_tiles": float(total_tiles),
        "num_blocks": float(num_blocks),
        "padding_ratio": float(scheduled_pairs.size(0) / max(total_pairs, 1)),
        "fill_ratio": float(valid_pairs / max(scheduled_pairs.size(0), 1)),
        "num_positions": float(num_positions),
        "hole_count": float(hole_count),
        "stage1_hole_count": float(stage1_hole_count),
        "stage1_fill_ratio": float(stage1_valid_pairs / max(stage1_pairs.size(0), 1))
        if stage1_pairs.numel() > 0 else 0.0,
        "stage1_num_tiles": float(stage1_num_tiles),
        "tail_num_tiles": float(total_tiles - stage1_num_tiles),
        "num_rounds": 1.0,
        "stage1_owner_pairs": float(stage1_owner_pairs),
        "stage1_owner_pair_fraction": float(stage1_owner_pairs / max(total_pairs, 1)),
        "stage1_owner_slots": float(stage1_owner_slots),
        "stage1_owner_atomic_saved": float(stage1_owner_pairs - stage1_owner_slots),
        "stage1_owner_atomic_saved_fraction": float(
            (stage1_owner_pairs - stage1_owner_slots) / max(total_pairs, 1)
        ),
        "stage1_refilled_pairs": float(stage1_refilled_pairs),
        "stage1_refilled_pair_fraction": float(stage1_refilled_pairs / max(total_pairs, 1)),
        "stage1_holes_pre_refill": float(stage1_holes_pre_refill),
        "stage1_pre_refill_fill_ratio": float(
            stage1_owner_pairs / max(stage1_num_tiles * BM, 1)
        ) if stage1_num_tiles > 0 else 0.0,
        "stage1_refill_hole_utilization": float(
            stage1_refilled_pairs / max(stage1_holes_pre_refill, 1)
        ) if stage1_holes_pre_refill > 0 else 0.0,
        "num_bundles": float(selected_bundles),
        "avg_union_size": float(union_sizes.float().mean().item()) if union_sizes.numel() > 0 else 0.0,
        "max_union_size": float(union_sizes.max().item()) if union_sizes.numel() > 0 else 0.0,
        "offset_quota_total_tiles": float(((offset_counts.to(torch.int64) + BM - 1) // BM).sum().item()),
        "candidate_stage1_tiles": float(candidate_stage1_tiles),
        "pruned_stage1_tiles": float(max(candidate_stage1_tiles - stage1_num_tiles, 0)),
        "pruned_stage1_tile_fraction": float(
            max(candidate_stage1_tiles - stage1_num_tiles, 0) / max(candidate_stage1_tiles, 1)
        ),
        "tail_pairs_after_refill": float(
            total_pairs - stage1_owner_pairs - stage1_refilled_pairs
        ),
        "cap_mode": "prune_holes",
        "selected_bundles": float(selected_bundles),
        "selected_bundle_fraction": float(selected_bundles / max(num_full_bundles, 1)),
    }
    return {
        "scheduled_pairs_cpu": scheduled_pairs.to(pair_dtype) if scheduled_pairs.dtype != pair_dtype else scheduled_pairs,
        "tile_offsets_cpu": tile_offsets.to(offset_dtype) if tile_offsets.dtype != offset_dtype else tile_offsets,
        "tile_keep_cpu": torch.empty((0,), dtype=torch.uint8, device=triplets.device),
        "stage1_pairs_cpu": stage1_pairs.to(pair_dtype) if stage1_pairs.dtype != pair_dtype else stage1_pairs,
        "tile_group_ids_cpu": tile_bundle_ids,
        "tile_bundle_ids_cpu": tile_bundle_ids,
        "final_tile_is_tail_cpu": final_tile_is_tail,
        "bundle_owner_rows_cpu": bundle_owner_rows.to(pair_dtype) if bundle_owner_rows.dtype != pair_dtype else bundle_owner_rows,
        "stats": stats,
    }


def _build_position_bundle_union_prune_holes_owner_cuda_runtime_fast(
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    num_rows: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    """Runtime-only custom-CUDA builder without debug metadata/statistics."""
    if (
        not pairs.is_cuda
        or pairs.dtype != torch.int32
        or offset_counts.dtype != torch.int32
        or pairs.dim() != 2
        or pairs.size(1) != 2
        or not hasattr(_C, "gtsparse3d_build_bundle_union_prune_holes_owner_runtime_cuda")
    ):
        return None

    scheduled_pairs, tile_offsets, tile_slot_states, bundle_tile_counts, total_tiles, selected_bundles = (
        _C.gtsparse3d_build_bundle_union_prune_holes_owner_runtime_cuda(
            pairs,
            offset_counts,
            BM,
            num_blocks,
            num_rows,
        )
    )
    return {
        "scheduled_pairs": scheduled_pairs.to(pair_dtype) if scheduled_pairs.dtype != pair_dtype else scheduled_pairs,
        "tile_offsets": tile_offsets.to(offset_dtype) if tile_offsets.dtype != offset_dtype else tile_offsets,
        "tile_slot_states": tile_slot_states,
        "bundle_tile_counts": bundle_tile_counts,
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }


def _build_position_bundle_union_prune_holes_owner_from_rows_cuda_runtime_fast(
    row_inputs: torch.Tensor,
    row_masks: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    global_offset_support: torch.Tensor | None = None,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    """Runtime-only custom-CUDA builder from compact row_inputs/row_masks."""
    if (
        not row_inputs.is_cuda
        or row_inputs.dtype != torch.int32
        or row_inputs.dim() != 2
        or row_masks.dtype != torch.int32
        or row_masks.dim() != 1
        or row_masks.size(0) != row_inputs.size(0)
        or offset_counts.dtype != torch.int32
        or offset_counts.dim() != 1
        or row_inputs.size(1) != offset_counts.numel()
    ):
        return None

    if global_offset_support is not None:
        if (
            global_offset_support.dtype != torch.int32
            or global_offset_support.dim() != 1
            or global_offset_support.numel() != offset_counts.numel()
            or not hasattr(_C, "gtsparse3d_build_bundle_union_prune_holes_owner_runtime_from_rows_with_support_cuda")
        ):
            return None
        scheduled_pairs, tile_offsets, tile_slot_states, bundle_tile_counts, total_tiles, selected_bundles = (
            _C.gtsparse3d_build_bundle_union_prune_holes_owner_runtime_from_rows_with_support_cuda(
                row_inputs,
                row_masks,
                global_offset_support,
                offset_counts,
                BM,
                num_blocks,
            )
        )
    else:
        if not hasattr(_C, "gtsparse3d_build_bundle_union_prune_holes_owner_runtime_from_rows_cuda"):
            return None
        scheduled_pairs, tile_offsets, tile_slot_states, bundle_tile_counts, total_tiles, selected_bundles = (
            _C.gtsparse3d_build_bundle_union_prune_holes_owner_runtime_from_rows_cuda(
                row_inputs,
                row_masks,
                offset_counts,
                BM,
                num_blocks,
            )
        )
    return {
        "scheduled_pairs": scheduled_pairs.to(pair_dtype) if scheduled_pairs.dtype != pair_dtype else scheduled_pairs,
        "tile_offsets": tile_offsets.to(offset_dtype) if tile_offsets.dtype != offset_dtype else tile_offsets,
        "tile_slot_states": tile_slot_states,
        "bundle_tile_counts": bundle_tile_counts,
        "total_tiles": total_tiles,
        "selected_bundles": selected_bundles,
    }


def _build_position_bundle_union_schedule_cpu_debug(
    triplets_cpu: torch.Tensor,
    offset_counts_cpu: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    cap_mode: str,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Debug builder for BM-position bundles with union offsets and eager hole refill."""
    K_vol = int(offset_counts_cpu.numel())
    total_pairs = int(offset_counts_cpu.sum().item())

    unique_out_rows, inverse = torch.unique(
        triplets_cpu[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets_cpu.dtype)
    inputs_by_pos_and_offset[inverse, triplets_cpu[:, 2].long()] = triplets_cpu[:, 0]
    active = inputs_by_pos_and_offset.ge(0)
    base_sorted_positions = _sort_positions_by_active_pattern(
        active, unique_out_rows.tolist())
    residual_active = active.clone()
    global_offset_support = active.sum(dim=0, dtype=torch.int64)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    tile_pairs_list: list[list[list[int]]] = []
    tile_offsets_list: list[int] = []
    tile_group_ids_list: list[int] = []
    bundle_owner_rows: list[list[int]] = []
    bundle_union_sizes: list[int] = []
    next_group_id = 0
    stage1_owner_pairs = 0
    stage1_owner_slots = 0
    offset_tile_quota = [int((int(offset_counts_cpu[off].item()) + BM - 1) // BM) for off in range(K_vol)]
    emitted_tiles_per_off = [0] * K_vol
    selected_bundles = 0

    if cap_mode not in {"none", "tile", "bundle", "prune_holes"}:
        raise ValueError(f"unknown bundle-union cap_mode: {cap_mode}")

    num_full_bundles = len(base_sorted_positions) // BM
    candidate_bundles: list[dict[str, object]] = []
    candidate_stage1_tiles = 0
    for bundle_idx in range(num_full_bundles):
        positions = base_sorted_positions[bundle_idx * BM: (bundle_idx + 1) * BM]
        if len(positions) != BM:
            continue
        group_active = active[positions]
        ordered_offsets = _ordered_offsets_by_group_support_cpu(
            group_active,
            global_offset_support,
        )
        if not ordered_offsets:
            continue
        owner_pairs = int(group_active.sum().item())
        union_size = len(ordered_offsets)
        saved_per_tile = float(owner_pairs - BM) / max(union_size, 1)
        owner_rows = [int(unique_out_rows[pos].item()) for pos in positions]
        offset_records: list[dict[str, object]] = []
        for off in ordered_offsets:
            tile_pairs: list[list[int]] = []
            fill_count = 0
            for pos, owner_out_row in zip(positions, owner_rows):
                input_row = int(inputs_by_pos_and_offset[pos, off].item())
                if input_row >= 0:
                    tile_pairs.append([input_row, owner_out_row])
                    fill_count += 1
                else:
                    tile_pairs.append([-1, owner_out_row])
            offset_records.append(
                {
                    "off": int(off),
                    "fill_count": fill_count,
                    "tile_pairs": tile_pairs,
                }
            )
        candidate_bundles.append(
            {
                "bundle_idx": bundle_idx,
                "positions": positions,
                "ordered_offsets": ordered_offsets,
                "owner_rows": owner_rows,
                "offset_records": offset_records,
                "owner_pairs": owner_pairs,
                "union_size": union_size,
                "saved_per_tile": saved_per_tile,
            }
        )
        candidate_stage1_tiles += union_size

    emitted_tiles_per_off = [0] * K_vol
    kept_offsets_by_bundle: dict[int, set[int]] = {}
    bundle_emit_order = candidate_bundles

    if cap_mode == "bundle":
        bundle_emit_order = sorted(
            candidate_bundles,
            key=lambda item: (
                -float(item["saved_per_tile"]),
                -int(item["owner_pairs"]),
                int(item["union_size"]),
                int(item["bundle_idx"]),
            ),
        )
        for bundle in bundle_emit_order:
            offset_records = bundle["offset_records"]
            offs = [int(record["off"]) for record in offset_records]
            if any(emitted_tiles_per_off[off] >= offset_tile_quota[off] for off in offs):
                continue
            kept_offsets_by_bundle[int(bundle["bundle_idx"])] = set(offs)
            for off in offs:
                emitted_tiles_per_off[off] += 1
    elif cap_mode == "prune_holes":
        per_offset_candidates: list[list[tuple[int, int, int, int]]] = [[] for _ in range(K_vol)]
        for bundle in candidate_bundles:
            bundle_idx = int(bundle["bundle_idx"])
            owner_pairs = int(bundle["owner_pairs"])
            for offset_order, record in enumerate(bundle["offset_records"]):
                per_offset_candidates[int(record["off"])].append(
                    (
                        int(record["fill_count"]),
                        owner_pairs,
                        bundle_idx,
                        offset_order,
                    )
                )
        for off, candidates in enumerate(per_offset_candidates):
            if not candidates:
                continue
            quota = offset_tile_quota[off]
            candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
            for fill_count, _owner_pairs, bundle_idx, _offset_order in candidates[:quota]:
                if fill_count <= 0:
                    continue
                kept_offsets_by_bundle.setdefault(bundle_idx, set()).add(off)
                emitted_tiles_per_off[off] += 1
    else:
        for bundle in candidate_bundles:
            bundle_idx = int(bundle["bundle_idx"])
            offset_records = bundle["offset_records"]
            if cap_mode == "none":
                kept_offsets = [int(record["off"]) for record in offset_records]
            else:
                kept_offsets = [
                    int(record["off"])
                    for record in offset_records
                    if emitted_tiles_per_off[int(record["off"])] < offset_tile_quota[int(record["off"])]
                ]
                for off in kept_offsets:
                    emitted_tiles_per_off[off] += 1
            if kept_offsets:
                kept_offsets_by_bundle[bundle_idx] = set(kept_offsets)

    for bundle in bundle_emit_order:
        bundle_idx = int(bundle["bundle_idx"])
        kept_offsets = kept_offsets_by_bundle.get(bundle_idx)
        if not kept_offsets:
            continue

        positions = bundle["positions"]
        owner_rows = bundle["owner_rows"]
        kept_records = [
            record for record in bundle["offset_records"]
            if int(record["off"]) in kept_offsets
        ]
        if not kept_records:
            continue

        group_id = next_group_id
        next_group_id += 1
        selected_bundles += 1
        bundle_owner_rows.append(list(owner_rows))
        bundle_union_sizes.append(len(kept_records))
        slot_owner_kept = [False] * BM

        group_tile_ids: list[int] = []
        for record in kept_records:
            off = int(record["off"])
            tile_id = len(tile_pairs_list)
            tile_offsets_list.append(off)
            tile_group_ids_list.append(group_id)
            tile_pairs = [pair[:] for pair in record["tile_pairs"]]
            fill_count = int(record["fill_count"])
            stage1_owner_pairs += fill_count
            for slot, pos in enumerate(positions):
                if tile_pairs[slot][0] < 0:
                    continue
                residual_active[pos, off] = False
                slot_owner_kept[slot] = True
            tile_pairs_list.append(tile_pairs)
            group_tile_ids.append(tile_id)

        stage1_owner_slots += sum(slot_owner_kept)
        target_col = min(range(num_blocks), key=column_lengths.__getitem__)
        columns[target_col].extend(group_tile_ids)
        column_lengths[target_col] += len(group_tile_ids)

    stage1_num_tiles = len(tile_pairs_list)
    stage1_holes_pre_refill = max(stage1_num_tiles * BM - stage1_owner_pairs, 0)

    fill_order = _sort_positions_by_active_pattern(
        residual_active, unique_out_rows.tolist())
    fill_queues = [
        deque(pos for pos in fill_order if bool(residual_active[pos, off].item()))
        for off in range(K_vol)
    ]
    stage1_refilled_pairs = 0
    for tile_id in range(stage1_num_tiles):
        off = tile_offsets_list[tile_id]
        queue = fill_queues[off]
        tile_pairs = tile_pairs_list[tile_id]
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
                    "bundle-union refill encountered an invalid residual pair; schedule invariant broken")
            tile_pairs[slot] = [input_row, int(unique_out_rows[pos].item())]
            residual_active[pos, off] = False
            stage1_refilled_pairs += 1

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
                        "bundle-union tail encountered an invalid residual pair; schedule invariant broken")
                tile_pairs[slot] = [input_row, int(unique_out_rows[pos].item())]
                residual_active[pos, off] = False
            tile_pairs_list.append(tile_pairs)

            target_col = min(range(num_blocks), key=column_lengths.__getitem__)
            columns[target_col].append(tile_id)
            column_lengths[target_col] += 1

    if residual_active.any():
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"bundle-union schedule left residual pairs after tail fill: {leftover_pairs}")

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
    stats = dict(debug["stats"])
    num_bundles = len(bundle_union_sizes)
    stats["stage1_owner_pairs"] = float(stage1_owner_pairs)
    stats["stage1_owner_pair_fraction"] = float(stage1_owner_pairs / max(total_pairs, 1))
    stats["stage1_owner_slots"] = float(stage1_owner_slots)
    stats["stage1_owner_atomic_saved"] = float(stage1_owner_pairs - stage1_owner_slots)
    stats["stage1_owner_atomic_saved_fraction"] = float(
        (stage1_owner_pairs - stage1_owner_slots) / max(total_pairs, 1)
    )
    stats["stage1_refilled_pairs"] = float(stage1_refilled_pairs)
    stats["stage1_refilled_pair_fraction"] = float(stage1_refilled_pairs / max(total_pairs, 1))
    stats["stage1_holes_pre_refill"] = float(stage1_holes_pre_refill)
    stats["stage1_pre_refill_fill_ratio"] = float(
        stage1_owner_pairs / max(stage1_num_tiles * BM, 1)
    ) if stage1_num_tiles > 0 else 0.0
    stats["stage1_refill_hole_utilization"] = float(
        stage1_refilled_pairs / max(stage1_holes_pre_refill, 1)
    ) if stage1_holes_pre_refill > 0 else 0.0
    stats["num_bundles"] = float(num_bundles)
    stats["avg_union_size"] = float(sum(bundle_union_sizes) / max(num_bundles, 1)) if num_bundles > 0 else 0.0
    stats["max_union_size"] = float(max(bundle_union_sizes)) if bundle_union_sizes else 0.0
    stats["offset_quota_total_tiles"] = float(sum(offset_tile_quota))
    stats["candidate_stage1_tiles"] = float(candidate_stage1_tiles)
    stats["pruned_stage1_tiles"] = float(max(candidate_stage1_tiles - stage1_num_tiles, 0))
    stats["pruned_stage1_tile_fraction"] = float(
        max(candidate_stage1_tiles - stage1_num_tiles, 0) / max(candidate_stage1_tiles, 1)
    )
    stats["tail_pairs_after_refill"] = float(
        total_pairs - stage1_owner_pairs - stage1_refilled_pairs
    )
    stats["cap_mode"] = cap_mode
    stats["selected_bundles"] = float(selected_bundles)
    stats["selected_bundle_fraction"] = float(selected_bundles / max(num_full_bundles, 1))
    debug["stats"] = stats
    debug["bundle_owner_rows_cpu"] = (
        torch.tensor(bundle_owner_rows, dtype=pair_dtype)
        if bundle_owner_rows
        else torch.empty((0, BM), dtype=pair_dtype)
    )
    return debug


def _build_position_bundle_union_schedule_gpu_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype,
    offset_dtype: torch.dtype,
    cap_mode: str,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """GPU-oriented bundle-union builder for quota/prune_holes variants.

    The heavy tensor work stays on device; only bundle/column bookkeeping remains on host.
    """
    if cap_mode not in {"tile", "prune_holes"}:
        raise ValueError(f"unsupported GPU bundle-union cap_mode: {cap_mode}")

    device = triplets.device
    K_vol = int(offset_counts.numel())
    total_pairs = int(offset_counts.sum().item())
    bit_values = (1 << torch.arange(K_vol, dtype=torch.int64, device=device))

    unique_out_rows, inverse = torch.unique(
        triplets[:, 1], sorted=True, return_inverse=True)
    num_positions = int(unique_out_rows.numel())
    out_rows = unique_out_rows.to(dtype=torch.int64)
    inputs_by_pos_and_offset = torch.full(
        (num_positions, K_vol), -1, dtype=triplets.dtype, device=device)
    inputs_by_pos_and_offset[inverse, triplets[:, 2].long()] = triplets[:, 0]
    active = inputs_by_pos_and_offset.ge(0)
    popcount = active.sum(dim=1, dtype=torch.int64)
    masks = (active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    base_sorted_positions = _sort_positions_by_mask_key_tensor(
        masks, popcount, out_rows, K_vol)
    residual_active = active.clone()
    global_offset_support = active.sum(dim=0, dtype=torch.int64)

    columns: list[list[int]] = [[] for _ in range(num_blocks)]
    column_lengths = [0] * num_blocks
    bundle_owner_rows: torch.Tensor
    bundle_union_sizes: torch.Tensor

    num_full_bundles = int(base_sorted_positions.numel()) // BM
    bundle_positions = base_sorted_positions[: num_full_bundles * BM].view(num_full_bundles, BM)
    owner_rows_matrix = unique_out_rows[bundle_positions].to(pair_dtype)
    group_active = active[bundle_positions]
    group_support = group_active.sum(dim=1, dtype=torch.int64)
    owner_pairs = group_support.sum(dim=1)
    candidate_tile_mask = group_support > 0
    candidate_stage1_tiles = int(candidate_tile_mask.sum().item())

    max_global = int(global_offset_support.max().item()) if K_vol > 0 else 0
    order_offsets = torch.arange(K_vol, dtype=torch.int64, device=device).view(1, -1)
    sort_key = (
        (BM + 1 - group_support.clamp(min=0, max=BM)) * ((max_global + 1) * max(K_vol, 1))
        + (max_global - global_offset_support.view(1, -1)) * max(K_vol, 1)
        + order_offsets
    )
    order_all = torch.argsort(sort_key, dim=1, stable=True)
    offset_rank = torch.empty_like(order_all)
    offset_rank.scatter_(
        1,
        order_all,
        torch.arange(K_vol, dtype=torch.int64, device=device).view(1, -1).expand(num_full_bundles, -1),
    )

    quota = ((offset_counts.to(torch.int64) + BM - 1) // BM).tolist()
    kept_mask = torch.zeros((num_full_bundles, K_vol), dtype=torch.bool, device=device)
    bundle_index = torch.arange(num_full_bundles, dtype=torch.int64, device=device)
    max_owner_pairs = int(owner_pairs.max().item()) if num_full_bundles > 0 else 0

    for off in range(K_vol):
        fill = group_support[:, off]
        valid = fill > 0
        if not bool(valid.any().item()):
            continue
        valid_bundle_idx = bundle_index[valid]
        if cap_mode == "tile":
            order = torch.argsort(
                valid_bundle_idx * max(K_vol, 1) + offset_rank[valid, off],
                stable=True,
            )
        else:
            fill_key = BM - fill[valid]
            owner_key = max_owner_pairs - owner_pairs[valid]
            order = torch.argsort(
                (((fill_key * (max_owner_pairs + 1)) + owner_key) * max(num_full_bundles, 1) + valid_bundle_idx)
                * max(K_vol, 1)
                + offset_rank[valid, off],
                stable=True,
            )
        chosen = valid_bundle_idx[order[: quota[off]]]
        if chosen.numel() > 0:
            kept_mask[chosen, off] = True

    keep_count = kept_mask.sum(dim=1, dtype=torch.int64)
    selected_bundle_mask = keep_count > 0
    selected_bundle_idx = torch.nonzero(selected_bundle_mask, as_tuple=False).view(-1)
    selected_bundles = int(selected_bundle_idx.numel())
    bundle_group_ids = torch.full((num_full_bundles,), -1, dtype=torch.int32, device=device)
    if selected_bundles > 0:
        bundle_group_ids[selected_bundle_idx] = torch.arange(
            selected_bundles, dtype=torch.int32, device=device)

    bundle_union_sizes = keep_count[selected_bundle_idx].to(torch.int64)
    bundle_owner_rows = owner_rows_matrix[selected_bundle_idx] if selected_bundles > 0 else torch.empty(
        (0, BM), dtype=pair_dtype, device=device)

    stage1_owner_pairs = int(group_support.masked_select(kept_mask).sum().item())
    kept_active = group_active[selected_bundle_idx] if selected_bundles > 0 else torch.empty(
        (0, BM, K_vol), dtype=torch.bool, device=device)
    kept_mask_selected = kept_mask[selected_bundle_idx] if selected_bundles > 0 else torch.empty(
        (0, K_vol), dtype=torch.bool, device=device)
    slot_owner_kept = (
        (kept_active & kept_mask_selected.unsqueeze(1)).any(dim=2)
        if selected_bundles > 0
        else torch.empty((0, BM), dtype=torch.bool, device=device)
    )
    stage1_owner_slots = int(slot_owner_kept.sum().item())

    stage1_bundle_ids_host: list[int] = []
    stage1_offsets_host: list[int] = []
    next_tile_id = 0
    for bundle_idx in range(num_full_bundles):
        keep_n = int(keep_count[bundle_idx].item())
        if keep_n <= 0:
            continue
        ordered_offsets = order_all[bundle_idx].tolist()
        kept_offsets = [
            off for off in ordered_offsets
            if bool(kept_mask[bundle_idx, off].item()) and int(group_support[bundle_idx, off].item()) > 0
        ]
        if not kept_offsets:
            continue
        group_id = int(bundle_group_ids[bundle_idx].item())
        target_col = min(range(num_blocks), key=column_lengths.__getitem__)
        group_tile_ids = list(range(next_tile_id, next_tile_id + len(kept_offsets)))
        columns[target_col].extend(group_tile_ids)
        column_lengths[target_col] += len(kept_offsets)
        stage1_bundle_ids_host.extend([bundle_idx] * len(kept_offsets))
        stage1_offsets_host.extend(kept_offsets)
        next_tile_id += len(kept_offsets)

    stage1_num_tiles = len(stage1_offsets_host)
    stage1_bundle_ids = torch.tensor(stage1_bundle_ids_host, dtype=torch.long, device=device)
    stage1_offsets = torch.tensor(stage1_offsets_host, dtype=torch.long, device=device)
    if stage1_num_tiles > 0:
        stage1_positions = bundle_positions[stage1_bundle_ids]
        flat_positions = stage1_positions.reshape(-1)
        flat_offsets = stage1_offsets.repeat_interleave(BM)
        stage1_inputs = inputs_by_pos_and_offset[flat_positions, flat_offsets].view(stage1_num_tiles, BM)
        stage1_rows = owner_rows_matrix[stage1_bundle_ids]
        stage1_pairs = torch.empty((stage1_num_tiles, BM, 2), dtype=pair_dtype, device=device)
        stage1_pairs[:, :, 0] = stage1_inputs.to(pair_dtype)
        stage1_pairs[:, :, 1] = stage1_rows

        valid_owner = stage1_inputs.reshape(-1) >= 0
        if bool(valid_owner.any().item()):
            owner_mask = torch.zeros_like(active)
            owner_mask[flat_positions[valid_owner], flat_offsets[valid_owner]] = True
            residual_active &= ~owner_mask
    else:
        stage1_pairs = torch.empty((0, BM, 2), dtype=pair_dtype, device=device)

    stage1_holes_pre_refill = max(stage1_num_tiles * BM - stage1_owner_pairs, 0)
    residual_masks = (residual_active.to(torch.int64) * bit_values.unsqueeze(0)).sum(dim=1)
    residual_popcount = residual_active.sum(dim=1, dtype=torch.int64)
    fill_order = _sort_positions_by_mask_key_tensor(
        residual_masks,
        residual_popcount,
        out_rows,
        K_vol,
    )

    stage1_refilled_pairs = 0
    for off in range(K_vol):
        tile_ids = torch.nonzero(stage1_offsets == off, as_tuple=False).view(-1)
        if tile_ids.numel() == 0:
            continue
        hole_mask = stage1_pairs[tile_ids, :, 0] < 0
        hole_count = int(hole_mask.sum().item())
        if hole_count <= 0:
            continue
        refill_positions = fill_order[residual_active[fill_order, off]]
        use = min(hole_count, int(refill_positions.numel()))
        if use <= 0:
            continue
        hole_linear = torch.nonzero(hole_mask.reshape(-1), as_tuple=False).view(-1)[:use]
        rows = hole_linear // BM
        slots = hole_linear % BM
        chosen_positions = refill_positions[:use]
        chosen_inputs = inputs_by_pos_and_offset[chosen_positions, off]
        stage1_pairs[tile_ids[rows], slots, 0] = chosen_inputs.to(pair_dtype)
        stage1_pairs[tile_ids[rows], slots, 1] = unique_out_rows[chosen_positions].to(pair_dtype)
        residual_active[chosen_positions, off] = False
        stage1_refilled_pairs += use

    tile_pair_batches: list[torch.Tensor] = []
    tile_offset_batches: list[torch.Tensor] = []
    tile_group_batches: list[torch.Tensor] = []
    total_tiles = 0
    next_group_id = selected_bundles

    if stage1_num_tiles > 0:
        tile_pair_batches.append(stage1_pairs)
        tile_offset_batches.append(stage1_offsets.to(dtype=offset_dtype))
        tile_group_batches.append(bundle_group_ids[stage1_bundle_ids])
        total_tiles = stage1_num_tiles

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
        tile_pairs_batch[tile_idx, tile_slot, 0] = inputs_by_pos_and_offset[leftover_positions, off].to(pair_dtype)
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

    if bool(residual_active.any().item()):
        leftover_pairs = int(residual_active.sum().item())
        raise RuntimeError(
            f"bundle-union GPU schedule left residual pairs after tail fill: {leftover_pairs}")

    debug = _finalize_schedule_from_tile_batches_gpu(
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
    stats = dict(debug["stats"])
    num_bundles = selected_bundles
    stats["stage1_owner_pairs"] = float(stage1_owner_pairs)
    stats["stage1_owner_pair_fraction"] = float(stage1_owner_pairs / max(total_pairs, 1))
    stats["stage1_owner_slots"] = float(stage1_owner_slots)
    stats["stage1_owner_atomic_saved"] = float(stage1_owner_pairs - stage1_owner_slots)
    stats["stage1_owner_atomic_saved_fraction"] = float(
        (stage1_owner_pairs - stage1_owner_slots) / max(total_pairs, 1)
    )
    stats["stage1_refilled_pairs"] = float(stage1_refilled_pairs)
    stats["stage1_refilled_pair_fraction"] = float(stage1_refilled_pairs / max(total_pairs, 1))
    stats["stage1_holes_pre_refill"] = float(stage1_holes_pre_refill)
    stats["stage1_pre_refill_fill_ratio"] = float(
        stage1_owner_pairs / max(stage1_num_tiles * BM, 1)
    ) if stage1_num_tiles > 0 else 0.0
    stats["stage1_refill_hole_utilization"] = float(
        stage1_refilled_pairs / max(stage1_holes_pre_refill, 1)
    ) if stage1_holes_pre_refill > 0 else 0.0
    stats["num_bundles"] = float(num_bundles)
    stats["avg_union_size"] = float(bundle_union_sizes.float().mean().item()) if num_bundles > 0 else 0.0
    stats["max_union_size"] = float(bundle_union_sizes.max().item()) if num_bundles > 0 else 0.0
    stats["offset_quota_total_tiles"] = float(((offset_counts.to(torch.int64) + BM - 1) // BM).sum().item())
    stats["candidate_stage1_tiles"] = float(candidate_stage1_tiles)
    stats["pruned_stage1_tiles"] = float(max(candidate_stage1_tiles - stage1_num_tiles, 0))
    stats["pruned_stage1_tile_fraction"] = float(
        max(candidate_stage1_tiles - stage1_num_tiles, 0) / max(candidate_stage1_tiles, 1)
    )
    stats["tail_pairs_after_refill"] = float(
        total_pairs - stage1_owner_pairs - stage1_refilled_pairs
    )
    stats["cap_mode"] = cap_mode
    stats["selected_bundles"] = float(selected_bundles)
    stats["selected_bundle_fraction"] = float(selected_bundles / max(num_full_bundles, 1))
    debug["stats"] = stats
    debug["bundle_owner_rows_cpu"] = bundle_owner_rows
    return debug


def build_position_bundle_union_schedule_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype | None = None,
    offset_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Public debug builder for the position-bundle-union experiment."""
    if pair_dtype is None:
        pair_dtype = triplets.dtype
    if offset_dtype is None:
        offset_dtype = offset_counts.dtype
    debug = _build_position_bundle_union_schedule_cpu_debug(
        triplets.detach().cpu(),
        offset_counts.detach().cpu(),
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        cap_mode="none",
    )
    return _move_schedule_debug_to_device(debug, triplets.device) if triplets.is_cuda else debug


def build_position_bundle_union_quota_schedule_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype | None = None,
    offset_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Bundle-union experiment with per-offset tile quotas capped at ceil(nnz/BM)."""
    if pair_dtype is None:
        pair_dtype = triplets.dtype
    if offset_dtype is None:
        offset_dtype = offset_counts.dtype
    if triplets.is_cuda:
        return _build_position_bundle_union_schedule_gpu_debug(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            cap_mode="tile",
        )
    debug = _build_position_bundle_union_schedule_cpu_debug(
        triplets.detach().cpu(),
        offset_counts.detach().cpu(),
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        cap_mode="tile",
    )
    return _move_schedule_debug_to_device(debug, triplets.device) if triplets.is_cuda else debug


def build_position_bundle_union_bundle_quota_schedule_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype | None = None,
    offset_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Bundle-union experiment with all-or-nothing bundle selection under per-offset quotas."""
    if pair_dtype is None:
        pair_dtype = triplets.dtype
    if offset_dtype is None:
        offset_dtype = offset_counts.dtype
    debug = _build_position_bundle_union_schedule_cpu_debug(
        triplets.detach().cpu(),
        offset_counts.detach().cpu(),
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        cap_mode="bundle",
    )
    return _move_schedule_debug_to_device(debug, triplets.device) if triplets.is_cuda else debug


def build_position_bundle_union_prune_holes_schedule_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype | None = None,
    offset_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Bundle-union experiment that keeps the fullest per-offset tiles and recycles hole-heavy tiles to tail."""
    if pair_dtype is None:
        pair_dtype = triplets.dtype
    if offset_dtype is None:
        offset_dtype = offset_counts.dtype
    if triplets.is_cuda:
        return _build_position_bundle_union_schedule_gpu_debug(
            triplets,
            offset_counts,
            BM,
            num_blocks,
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
            cap_mode="prune_holes",
        )
    debug = _build_position_bundle_union_schedule_cpu_debug(
        triplets.detach().cpu(),
        offset_counts.detach().cpu(),
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
        cap_mode="prune_holes",
    )
    return _move_schedule_debug_to_device(debug, triplets.device) if triplets.is_cuda else debug


def build_position_bundle_union_prune_holes_owner_schedule_debug(
    triplets: torch.Tensor,
    offset_counts: torch.Tensor,
    BM: int,
    num_blocks: int,
    *,
    pair_dtype: torch.dtype | None = None,
    offset_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor | dict[str, float]]:
    """Return prune_holes schedule plus final ordered bundle ids for the owner-bank kernel."""
    if pair_dtype is None:
        pair_dtype = triplets.dtype
    if offset_dtype is None:
        offset_dtype = offset_counts.dtype
    fast = _build_position_bundle_union_prune_holes_owner_cuda_fast(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )
    if fast is not None:
        return fast

    debug = build_position_bundle_union_prune_holes_schedule_debug(
        triplets,
        offset_counts,
        BM,
        num_blocks,
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )
    tile_bundle_ids = debug["tile_group_ids_cpu"].clone()
    if tile_bundle_ids.numel() > 0:
        tile_bundle_ids = tile_bundle_ids.to(dtype=torch.int32)
        tile_bundle_ids[debug["final_tile_is_tail_cpu"]] = -1
    debug["tile_bundle_ids_cpu"] = tile_bundle_ids
    return debug


def compare_position_bundle_union_vs_subset_first(
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    C_out: int,
    config_id: int,
    reuse_mode: str = "row_selective",
) -> dict[str, dict[str, float] | str]:
    """Compare the bundle-union experiment against the existing subset-first schedule."""
    cfg = get_scheduled_simt_config(config_id, reuse_mode=reuse_mode)
    launch_info = scheduled_expanded_worklist_launch_info(config_id, reuse_mode=reuse_mode)
    triplets, offset_starts = compact_triplet_worklist(pairs, offset_counts)
    pair_dtype = triplets.dtype
    offset_dtype = offset_counts.dtype

    if triplets.is_cuda:
        subset_debug = _build_minimal_scheduled_worklist_cpu_debug(
            triplets.detach().cpu(),
            offset_counts.detach().cpu(),
            cfg["BM"],
            launch_info["grid_dim_x"],
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    else:
        subset_debug = _build_minimal_scheduled_worklist_cpu_debug(
            triplets,
            offset_counts,
            cfg["BM"],
            launch_info["grid_dim_x"],
            pair_dtype=pair_dtype,
            offset_dtype=offset_dtype,
        )
    bundle_debug = build_position_bundle_union_schedule_debug(
        triplets,
        offset_counts,
        cfg["BM"],
        launch_info["grid_dim_x"],
        pair_dtype=pair_dtype,
        offset_dtype=offset_dtype,
    )

    baseline_pairs, _ = build_baseline_fixed_width_tiles_from_triplets(
        triplets,
        offset_counts,
        offset_starts,
        BM=cfg["BM"],
    )
    num_bn_tiles = (C_out + cfg["BN"] - 1) // cfg["BN"]
    baseline = evaluate_bn_outer_row_locality(
        baseline_pairs,
        num_blocks=launch_info["grid_dim_x"],
        BM=cfg["BM"],
        num_bn_tiles=num_bn_tiles,
    )

    def _summarize_debug(
        debug: dict[str, torch.Tensor | dict[str, float]],
    ) -> dict[str, float]:
        stage1 = evaluate_bn_outer_row_locality(
            debug["stage1_pairs_cpu"],
            num_blocks=launch_info["grid_dim_x"],
            BM=cfg["BM"],
            num_bn_tiles=num_bn_tiles,
            tile_group_ids=debug["tile_group_ids_cpu"],
        )
        final = evaluate_bn_outer_row_locality(
            debug["scheduled_pairs_cpu"],
            num_blocks=launch_info["grid_dim_x"],
            BM=cfg["BM"],
            num_bn_tiles=num_bn_tiles,
            tile_group_ids=debug["tile_group_ids_cpu"],
        )
        stats = debug["stats"]
        return {
            "padding_ratio": float(stats["padding_ratio"]),
            "fill_ratio": float(stats["fill_ratio"]),
            "stage1_fill_ratio": float(stats["stage1_fill_ratio"]),
            "stage1_same_slot_rate": float(stage1["same_slot_rate"]),
            "stage1_any_hit_rate": float(stage1["any_hit_rate"]),
            "stage1_exact_tile_rate": float(stage1["exact_tile_rate"]),
            "scheduled_same_slot_rate": float(final["same_slot_rate"]),
            "scheduled_any_hit_rate": float(final["any_hit_rate"]),
            "scheduled_exact_tile_rate": float(final["exact_tile_rate"]),
            "stage1_num_tiles": float(stats["stage1_num_tiles"]),
            "tail_num_tiles": float(stats["tail_num_tiles"]),
            "stage1_owner_pair_fraction": float(stats.get("stage1_owner_pair_fraction", 0.0)),
            "stage1_refilled_pair_fraction": float(stats.get("stage1_refilled_pair_fraction", 0.0)),
            "avg_union_size": float(stats.get("avg_union_size", 0.0)),
            "max_union_size": float(stats.get("max_union_size", 0.0)),
        }

    subset = _summarize_debug(subset_debug)
    bundle = _summarize_debug(bundle_debug)
    return {
        "variant": "position_bundle_union",
        "baseline": {
            "same_slot_rate": float(baseline["same_slot_rate"]),
            "any_hit_rate": float(baseline["any_hit_rate"]),
            "exact_tile_rate": float(baseline["exact_tile_rate"]),
        },
        "subset_first": subset,
        "position_bundle_union": bundle,
        "delta": {
            "stage1_same_slot_rate": bundle["stage1_same_slot_rate"] - subset["stage1_same_slot_rate"],
            "scheduled_same_slot_rate": bundle["scheduled_same_slot_rate"] - subset["scheduled_same_slot_rate"],
            "padding_ratio": bundle["padding_ratio"] - subset["padding_ratio"],
            "stage1_fill_ratio": bundle["stage1_fill_ratio"] - subset["stage1_fill_ratio"],
            "stage1_num_tiles": bundle["stage1_num_tiles"] - subset["stage1_num_tiles"],
        },
    }
