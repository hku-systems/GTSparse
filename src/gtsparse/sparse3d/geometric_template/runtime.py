from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from gtsparse import _C
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

BM = 128
CENTER_OFFSET = 13
NUM_LOGICAL_OFFSETS = 27
FP32_SETTING_BK = {1: 16, 2: 32, 3: 32}
FP32_SETTING_BN = {1: 16, 2: 16, 3: 64}
FP16_SETTING_BK = {1: 16, 2: 32, 3: 32}
FP16_SETTING_BN = {1: 16, 2: 16, 3: 64}
RING_ORDER = (
    0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4,
    22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26,
)
MATCH_LOGICAL_TO_ACTUAL = (
    0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4,
    13, 22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26,
)
PAYLOAD_LOGICAL_TO_ACTUAL = (
    *RING_ORDER,
    CENTER_OFFSET,
)
MATCH_CENTER_SLOT = 13

TEMPLATE_CENTER = 0
TEMPLATE_SKIP2_KEEP0 = 1
TEMPLATE_SKIP2_KEEP1 = 2
TEMPLATE_SKIP2_KEEP2 = 3
TEMPLATE_SKIP1_HOLE0 = 4
TEMPLATE_SKIP1_HOLE1 = 5
TEMPLATE_SKIP1_HOLE2 = 6
TEMPLATE_FULL27 = 7
NUM_TEMPLATES = 8

PAYLOAD_WIDTH_W1 = 1
PAYLOAD_WIDTH_W9 = 10
PAYLOAD_WIDTH_W18 = 19
PAYLOAD_WIDTH_W27 = 27
EMPTY_TENSOR = torch.empty(0)


def _match_logical_to_actual_index(device: torch.device) -> torch.Tensor:
    return torch.tensor(MATCH_LOGICAL_TO_ACTUAL, device=device, dtype=torch.long)


def _payload_logical_to_actual_index(device: torch.device) -> torch.Tensor:
    return torch.tensor(PAYLOAD_LOGICAL_TO_ACTUAL, device=device, dtype=torch.long)


def _build_template_tables() -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    payload_slot_of_actual = {
        int(off): idx for idx, off in enumerate(PAYLOAD_LOGICAL_TO_ACTUAL)
    }

    def center_payload_slots() -> tuple[int, ...]:
        return (payload_slot_of_actual[CENTER_OFFSET],)

    def periodic_payload_slots(*, keep_residue: int | None = None, hole_residue: int | None = None) -> tuple[int, ...]:
        if (keep_residue is None) == (hole_residue is None):
            raise ValueError("exactly one of keep_residue or hole_residue must be set")
        if keep_residue is not None:
            kept_actual = {
                int(MATCH_LOGICAL_TO_ACTUAL[slot])
                for slot in range(NUM_LOGICAL_OFFSETS)
                if slot % 3 == int(keep_residue)
            }
        else:
            kept_actual = {
                int(MATCH_LOGICAL_TO_ACTUAL[slot])
                for slot in range(NUM_LOGICAL_OFFSETS)
                if slot % 3 != int(hole_residue)
            }
        slots = [
            payload_slot_of_actual[int(off)]
            for off in PAYLOAD_LOGICAL_TO_ACTUAL
            if int(off) in kept_actual and int(off) != CENTER_OFFSET
        ]
        slots.append(payload_slot_of_actual[CENTER_OFFSET])
        return tuple(slots)

    def periodic_reject_mask(*, keep_residue: int | None = None, hole_residue: int | None = None) -> tuple[int, ...]:
        mask: list[int] = []
        for slot in range(NUM_LOGICAL_OFFSETS):
            if slot == MATCH_CENTER_SLOT:
                mask.append(0)
            elif keep_residue is not None:
                mask.append(0 if (slot % 3) == int(keep_residue) else 1)
            elif hole_residue is not None:
                mask.append(0 if (slot % 3) != int(hole_residue) else 1)
            else:
                raise ValueError("periodic reject mask requires one residue selector")
        return tuple(mask)

    keep_slots = (
        center_payload_slots(),
        periodic_payload_slots(keep_residue=0),
        periodic_payload_slots(keep_residue=1),
        periodic_payload_slots(keep_residue=2),
        periodic_payload_slots(hole_residue=0),
        periodic_payload_slots(hole_residue=1),
        periodic_payload_slots(hole_residue=2),
        tuple(range(NUM_LOGICAL_OFFSETS)),
    )
    reject_masks = (
        tuple(0 if slot == MATCH_CENTER_SLOT else 1 for slot in range(NUM_LOGICAL_OFFSETS)),
        periodic_reject_mask(keep_residue=0),
        periodic_reject_mask(keep_residue=1),
        periodic_reject_mask(keep_residue=2),
        periodic_reject_mask(hole_residue=0),
        periodic_reject_mask(hole_residue=1),
        periodic_reject_mask(hole_residue=2),
        tuple(0 for _ in range(NUM_LOGICAL_OFFSETS)),
    )
    slot_counts = tuple(len(v) for v in keep_slots)
    reject_mask_ints = tuple(
        sum(int(bit) << idx for idx, bit in enumerate(mask))
        for mask in reject_masks
    )
    return keep_slots, reject_masks, slot_counts, reject_mask_ints


TEMPLATE_KEEP_SLOTS, TEMPLATE_REJECT_MASKS, TEMPLATE_SLOT_COUNTS, TEMPLATE_REJECT_MASKS_INT = _build_template_tables()


@dataclass(slots=True)
class GeometricTemplateRuntime:
    out_rows: torch.Tensor
    input_rows_w1: torch.Tensor
    input_rows_w9: torch.Tensor
    input_rows_w18: torch.Tensor
    input_rows_w27: torch.Tensor
    template_ids: torch.Tensor
    input_row_offsets: torch.Tensor
    template_counts: torch.Tensor
    padded_counts: torch.Tensor
    n_out: int
    bm: int = BM
    out_coords: Optional[torch.Tensor] = None
    out_spatial: Optional[Tuple[int, int, int]] = None
    coord_hashmap: Optional[torch.Tensor] = None
    reverse_dense_out_in_map: Optional[torch.Tensor] = None
    reverse_masks: Optional[torch.Tensor] = None
    sorted: bool = False


def _make_runtime(
    *,
    out_rows: torch.Tensor,
    input_rows_w1: torch.Tensor,
    input_rows_w9: torch.Tensor,
    input_rows_w18: torch.Tensor,
    input_rows_w27: torch.Tensor,
    template_ids: torch.Tensor,
    input_row_offsets: torch.Tensor,
    template_counts: torch.Tensor,
    padded_counts: torch.Tensor,
    n_out: int,
    bm: int,
    out_coords: Optional[torch.Tensor] = None,
    out_spatial: Optional[Tuple[int, int, int]] = None,
    coord_hashmap: Optional[torch.Tensor] = None,
    reverse_dense_out_in_map: Optional[torch.Tensor] = None,
    reverse_masks: Optional[torch.Tensor] = None,
    sorted: bool = False,
) -> GeometricTemplateRuntime:
    return GeometricTemplateRuntime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        template_counts=template_counts,
        padded_counts=padded_counts,
        n_out=n_out,
        bm=bm,
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=coord_hashmap,
        reverse_dense_out_in_map=reverse_dense_out_in_map,
        reverse_masks=reverse_masks,
        sorted=bool(sorted),
    )


def permute_weight_to_runtime_order(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype not in (torch.float16, torch.float32):
        raise TypeError("geometric template runtime requires float16 or float32 weight")
    if not weight.is_contiguous():
        raise ValueError("geometric template runtime requires contiguous weight for cache permutation")
    if weight.dim() == 2:
        if int(weight.size(0)) % NUM_LOGICAL_OFFSETS != 0:
            raise ValueError("2D weight must have first dimension divisible by 27")
        cin = int(weight.size(0)) // NUM_LOGICAL_OFFSETS
        cout = int(weight.size(1))
        weight_3d = weight.view(NUM_LOGICAL_OFFSETS, cin, cout)
    elif weight.dim() == 3:
        if int(weight.size(0)) != NUM_LOGICAL_OFFSETS:
            raise ValueError("3D weight must have shape [27, Cin, Cout]")
        cin = int(weight.size(1))
        cout = int(weight.size(2))
        weight_3d = weight
    else:
        raise ValueError("weight must be 2D flattened or 3D [27, Cin, Cout]")

    logical = weight_3d.index_select(0, _payload_logical_to_actual_index(weight.device)).contiguous()
    return logical.view(NUM_LOGICAL_OFFSETS * cin, cout)


def build_runtime_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
    sorted: bool = False,
) -> GeometricTemplateRuntime:
    del kernel_size, padding, dilation
    tensors = _C.gtsparse3d_finalize_row_template_center_last_build_runtime_from_coords(
        st.indices,
        max_bm,
        EMPTY_TENSOR if st.coord_hashmap is None else st.coord_hashmap,
        bool(sorted),
    )
    out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, template_counts, padded_counts, coord_hashmap = tensors
    return _make_runtime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        template_counts=template_counts,
        padded_counts=padded_counts,
        n_out=st.indices.size(0),
        bm=max_bm,
        out_coords=st.indices,
        out_spatial=tuple(st.spatial_shape),
        coord_hashmap=coord_hashmap,
        sorted=bool(sorted),
    )


def build_full_runtime_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
    lookup_coord_hashmap: Optional[torch.Tensor] = None,
    build_reverse_cache: bool = False,
    sorted: bool = False,
) -> GeometricTemplateRuntime:
    out_spatial = tuple(
        (st.spatial_shape[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
        for i in range(3)
    )
    effective_lookup_coord_hashmap = st.coord_hashmap if lookup_coord_hashmap is None else lookup_coord_hashmap
    tensors = _C.gtsparse3d_finalize_row_template_center_last_build_full_runtime_from_coords(
        st.indices,
        out_spatial[0],
        out_spatial[1],
        out_spatial[2],
        stride[0],
        stride[1],
        stride[2],
        padding[0],
        padding[1],
        padding[2],
        dilation[0],
        dilation[1],
        dilation[2],
        max_bm,
        EMPTY_TENSOR if effective_lookup_coord_hashmap is None else effective_lookup_coord_hashmap,
        build_reverse_cache,
        bool(sorted),
    )
    out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, template_counts, padded_counts, reverse_dense_out_in_map, reverse_masks, out_coords, coord_hashmap = tensors
    return _make_runtime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        template_counts=template_counts,
        padded_counts=padded_counts,
        n_out=out_coords.size(0),
        bm=max_bm,
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=coord_hashmap,
        reverse_dense_out_in_map=reverse_dense_out_in_map if reverse_dense_out_in_map.numel() else None,
        reverse_masks=reverse_masks if reverse_masks.numel() else None,
        sorted=bool(sorted),
    )


def build_reverse_runtime_from_full_runtime(
    forward_runtime: GeometricTemplateRuntime,
    out_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int] | list[int],
    *,
    max_bm: int,
    sorted: bool | None = None,
) -> GeometricTemplateRuntime:
    out_spatial = tuple(out_spatial)
    effective_sorted = bool(forward_runtime.sorted if sorted is None else sorted)
    if (
        forward_runtime.reverse_dense_out_in_map is not None
        and forward_runtime.reverse_masks is not None
        and forward_runtime.reverse_dense_out_in_map.size(0) == out_coords.size(0)
    ):
        tensors = _C.gtsparse3d_finalize_row_template_center_last_build_runtime_from_dense_out_in_map(
            forward_runtime.reverse_dense_out_in_map,
            forward_runtime.reverse_masks,
            max_bm,
            bool(effective_sorted),
        )
    else:
        tensors = _C.gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_from_full_runtime(
            forward_runtime.out_rows,
            forward_runtime.input_rows_w1,
            forward_runtime.input_rows_w9,
            forward_runtime.input_rows_w18,
            forward_runtime.input_rows_w27,
            forward_runtime.template_ids,
            forward_runtime.input_row_offsets,
            forward_runtime.padded_counts,
            out_coords.size(0),
            max_bm,
            bool(effective_sorted),
        )
    out_rows, input_rows_w1, input_rows_w9, input_rows_w18, input_rows_w27, template_ids, input_row_offsets, template_counts, padded_counts = tensors
    return _make_runtime(
        out_rows=out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        template_counts=template_counts,
        padded_counts=padded_counts,
        n_out=out_coords.size(0),
        bm=max_bm,
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=None,
        sorted=bool(effective_sorted),
    )


def fp32_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: GeometricTemplateRuntime,
    *,
    setting: int | None = None,
) -> torch.Tensor:
    if setting is None:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp32_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp32_forward
        )
    elif setting == 1:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp32_setting1_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp32_setting1_forward
        )
    elif setting == 2:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp32_setting2_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp32_setting2_forward
        )
    else:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp32_setting3_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp32_setting3_forward
        )
    return fn(
        features,
        logical_weight,
        runtime.out_rows,
        runtime.input_rows_w1,
        runtime.input_rows_w9,
        runtime.input_rows_w18,
        runtime.input_rows_w27,
        runtime.template_ids,
        runtime.input_row_offsets,
        runtime.n_out,
    )


def fp16_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: GeometricTemplateRuntime,
    *,
    setting: int | None = None,
) -> torch.Tensor:
    if setting is None:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp16_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp16_forward
        )
    elif setting == 1:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp16_setting1_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp16_setting1_forward
        )
    elif setting == 2:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp16_setting2_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp16_setting2_forward
        )
    else:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp16_setting3_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp16_setting3_forward
        )
    return fn(
        features,
        logical_weight,
        runtime.out_rows,
        runtime.input_rows_w1,
        runtime.input_rows_w9,
        runtime.input_rows_w18,
        runtime.input_rows_w27,
        runtime.template_ids,
        runtime.input_row_offsets,
        runtime.n_out,
    )
