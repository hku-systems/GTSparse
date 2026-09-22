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
TEMPLATE_KEEP_MASKS = tuple(
    tuple(int(i in keep) for i in range(NUM_LOGICAL_OFFSETS))
    for keep in TEMPLATE_KEEP_SLOTS
)


@dataclass(slots=True)
class FinalizedRowTemplateCenterLastRuntime:
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

    @property
    def template_stride(self) -> int:
        return int(self.input_rows_w1.size(1))

    @property
    def padded_rows(self) -> int:
        return int(self.out_rows.numel())

    @property
    def num_tiles(self) -> int:
        return self.padded_rows // int(self.bm)


def _from_cuda_runtime_tuple(
    tensors: tuple[torch.Tensor, ...],
    *,
    n_out: int,
    bm: int,
    out_coords: Optional[torch.Tensor] = None,
    out_spatial: Optional[Tuple[int, int, int]] = None,
    coord_hashmap: Optional[torch.Tensor] = None,
    reverse_dense_out_in_map: Optional[torch.Tensor] = None,
    reverse_masks: Optional[torch.Tensor] = None,
    sorted: bool = False,
) -> FinalizedRowTemplateCenterLastRuntime:
    return FinalizedRowTemplateCenterLastRuntime(
        out_rows=tensors[0],
        input_rows_w1=tensors[1],
        input_rows_w9=tensors[2],
        input_rows_w18=tensors[3],
        input_rows_w27=tensors[4],
        template_ids=tensors[5],
        input_row_offsets=tensors[6],
        template_counts=tensors[7],
        padded_counts=tensors[8],
        n_out=int(n_out),
        bm=int(bm),
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=coord_hashmap,
        reverse_dense_out_in_map=reverse_dense_out_in_map,
        reverse_masks=reverse_masks,
        sorted=bool(sorted),
    )


def permute_weight_to_center_last_order(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype not in (torch.float16, torch.float32):
        raise TypeError("center-last row-template path requires float16 or float32 weight")
    if not weight.is_contiguous():
        raise ValueError("center-last row-template path requires contiguous weight for cache permutation")
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


def build_center_last_runtime_from_dense_out_in_map(
    dense_out_in_map: torch.Tensor,
    *,
    out_rows: Optional[torch.Tensor] = None,
    bm: int = BM,
    sorted: bool = False,
) -> FinalizedRowTemplateCenterLastRuntime:
    if dense_out_in_map.dtype != torch.int32:
        raise TypeError("dense_out_in_map must be int32")
    if dense_out_in_map.dim() != 2 or int(dense_out_in_map.size(1)) != NUM_LOGICAL_OFFSETS:
        raise ValueError("dense_out_in_map must have shape [N_out, 27] in actual flatten order")
    if int(bm) != BM:
        raise ValueError("current center-last fp32 setting3 runtime requires bm=128")

    device = dense_out_in_map.device
    n_out = int(dense_out_in_map.size(0))
    if out_rows is None:
        out_rows = torch.arange(n_out, device=device, dtype=torch.int32)
    else:
        if out_rows.dtype != torch.int32:
            raise TypeError("out_rows must be int32")
        if out_rows.dim() != 1 or int(out_rows.numel()) != n_out:
            raise ValueError("out_rows must have shape [N_out]")
        if out_rows.device != device:
            raise ValueError("out_rows must be on the same device as dense_out_in_map")

    match_dense = dense_out_in_map.index_select(1, _match_logical_to_actual_index(device))
    payload_dense = dense_out_in_map.index_select(1, _payload_logical_to_actual_index(device))
    active = match_dense >= 0

    template_ids_src = torch.full((n_out,), -1, dtype=torch.int32, device=device)
    remaining = torch.ones((n_out,), dtype=torch.bool, device=device)

    for template_id, reject_mask_seq in enumerate(TEMPLATE_REJECT_MASKS):
        reject_mask = torch.tensor(reject_mask_seq, device=device, dtype=torch.bool)
        blocked = active & reject_mask.view(1, NUM_LOGICAL_OFFSETS)
        fits = remaining & (~blocked.any(dim=1))
        template_ids_src[fits] = int(template_id)
        remaining &= ~fits

    if bool(remaining.any().item()):
        raise RuntimeError("logical template builder left some rows unclassified")

    template_counts = torch.zeros((NUM_TEMPLATES,), dtype=torch.int32, device=device)
    padded_counts = torch.zeros((NUM_TEMPLATES,), dtype=torch.int32, device=device)
    rows_by_template: list[torch.Tensor] = []
    payload_by_template: list[torch.Tensor] = []
    template_stride = 0

    for template_id in range(NUM_TEMPLATES):
        rows = torch.nonzero(template_ids_src == template_id, as_tuple=False).flatten()
        count = int(rows.numel())
        padded = ((count + bm - 1) // bm) * bm
        template_counts[template_id] = count
        padded_counts[template_id] = padded
        template_stride = max(template_stride, padded)
        if count > 0:
            keep_slots = torch.tensor(TEMPLATE_KEEP_SLOTS[template_id], device=device, dtype=torch.long)
            payload = payload_dense.index_select(0, rows).index_select(1, keep_slots)
            if sorted and count > 1:
                active = payload >= 0
                bit_weights = (1 << torch.arange(int(payload.size(1)), device=device, dtype=torch.int64)).view(1, -1)
                mask = (active.to(torch.int64) * bit_weights).sum(dim=1)
                popcount = active.sum(dim=1, dtype=torch.int64)
                sort_key = (popcount << NUM_LOGICAL_OFFSETS) | mask
                order = torch.argsort(sort_key, descending=True)
                rows = rows.index_select(0, order)
                payload = payload.index_select(0, order)
        else:
            payload = torch.empty((0, 0), device=device, dtype=torch.int32)
        rows_by_template.append(rows)
        payload_by_template.append(payload)

    input_rows_w1 = torch.full(
        (1, template_stride, PAYLOAD_WIDTH_W1),
        -1,
        dtype=torch.int32,
        device=device,
    )
    input_rows_w9 = torch.full(
        (3, template_stride, PAYLOAD_WIDTH_W9),
        -1,
        dtype=torch.int32,
        device=device,
    )
    input_rows_w18 = torch.full(
        (3, template_stride, PAYLOAD_WIDTH_W18),
        -1,
        dtype=torch.int32,
        device=device,
    )
    input_rows_w27 = torch.full(
        (1, template_stride, PAYLOAD_WIDTH_W27),
        -1,
        dtype=torch.int32,
        device=device,
    )

    total_rows = int(padded_counts.sum().item())
    packed_out_rows = torch.full((total_rows,), -1, dtype=torch.int32, device=device)
    packed_template_ids = torch.full((total_rows,), -1, dtype=torch.int32, device=device)
    packed_input_row_offsets = torch.full((total_rows,), 0, dtype=torch.int32, device=device)

    global_base = 0
    for template_id, rows in enumerate(rows_by_template):
        count = int(rows.numel())
        padded = int(padded_counts[template_id].item())
        if padded == 0:
            continue

        packed_template_ids[global_base : global_base + padded] = int(template_id)
        packed_input_row_offsets[global_base : global_base + padded] = torch.arange(
            padded, device=device, dtype=torch.int32
        )

        if count > 0:
            packed_out_rows[global_base : global_base + count] = out_rows.index_select(0, rows)
            payload = payload_by_template[template_id]
            payload_width = int(payload.size(1))
            if template_id == TEMPLATE_CENTER:
                input_rows_w1[0, :count, :payload_width] = payload
            elif template_id <= TEMPLATE_SKIP2_KEEP2:
                input_rows_w9[template_id - TEMPLATE_SKIP2_KEEP0, :count, :payload_width] = payload
            elif template_id <= TEMPLATE_SKIP1_HOLE2:
                input_rows_w18[template_id - TEMPLATE_SKIP1_HOLE0, :count, :payload_width] = payload
            else:
                input_rows_w27[0, :count, :payload_width] = payload

        global_base += padded

    return FinalizedRowTemplateCenterLastRuntime(
        out_rows=packed_out_rows,
        input_rows_w1=input_rows_w1,
        input_rows_w9=input_rows_w9,
        input_rows_w18=input_rows_w18,
        input_rows_w27=input_rows_w27,
        template_ids=packed_template_ids,
        input_row_offsets=packed_input_row_offsets,
        template_counts=template_counts,
        padded_counts=padded_counts,
        n_out=n_out,
        bm=bm,
        sorted=bool(sorted),
    )


def build_center_last_runtime_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
    sorted: bool = False,
) -> FinalizedRowTemplateCenterLastRuntime:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("center-last builder currently requires kernel_size=(3, 3, 3)")
    if tuple(int(v) for v in padding) != (1, 1, 1):
        raise ValueError("center-last builder currently requires padding=(1, 1, 1)")
    if tuple(int(v) for v in dilation) != (1, 1, 1):
        raise ValueError("center-last builder currently requires dilation=(1, 1, 1)")
    tensors = _C.gtsparse3d_finalize_row_template_center_last_build_runtime_from_coords(
        st.indices.contiguous(),
        int(max_bm),
        torch.Tensor() if st.coord_hashmap is None else st.coord_hashmap,
        bool(sorted),
    )
    return _from_cuda_runtime_tuple(
        tensors[:9],
        n_out=int(st.indices.size(0)),
        bm=int(max_bm),
        out_coords=st.indices,
        out_spatial=tuple(int(v) for v in st.spatial_shape),
        coord_hashmap=tensors[9],
        sorted=bool(sorted),
    )


def build_center_last_full_runtime_from_coords(
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
) -> FinalizedRowTemplateCenterLastRuntime:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("center-last full builder currently requires kernel_size=(3, 3, 3)")
    stride = tuple(int(v) for v in stride)
    padding = tuple(int(v) for v in padding)
    dilation = tuple(int(v) for v in dilation)
    out_spatial = tuple(
        (int(st.spatial_shape[i]) + 2 * padding[i] - dilation[i] * (int(kernel_size[i]) - 1) - 1) // stride[i] + 1
        for i in range(3)
    )
    effective_lookup_coord_hashmap = st.coord_hashmap if lookup_coord_hashmap is None else lookup_coord_hashmap
    tensors = _C.gtsparse3d_finalize_row_template_center_last_build_full_runtime_from_coords(
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
        bool(build_reverse_cache),
        bool(sorted),
    )
    reverse_dense_out_in_map = tensors[9]
    reverse_masks = tensors[10]
    out_coords = tensors[11]
    return _from_cuda_runtime_tuple(
        tensors[:9],
        n_out=int(out_coords.size(0)),
        bm=int(max_bm),
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=tensors[12],
        reverse_dense_out_in_map=(reverse_dense_out_in_map if int(reverse_dense_out_in_map.numel()) > 0 else None),
        reverse_masks=(reverse_masks if int(reverse_masks.numel()) > 0 else None),
        sorted=bool(sorted),
    )


def build_center_last_reverse_runtime_from_coords(
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
    sorted: bool = False,
) -> FinalizedRowTemplateCenterLastRuntime:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("center-last reverse builder currently requires kernel_size=(3, 3, 3)")
    stride = tuple(int(v) for v in stride)
    padding = tuple(int(v) for v in padding)
    dilation = tuple(int(v) for v in dilation)
    out_spatial = tuple(int(v) for v in out_spatial)
    effective_lookup_coord_hashmap = lookup_st.coord_hashmap if lookup_coord_hashmap is None else lookup_coord_hashmap
    tensors = _C.gtsparse3d_finalize_row_template_center_last_build_reverse_runtime_from_coords(
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
        bool(sorted),
    )
    return _from_cuda_runtime_tuple(
        tensors[:9],
        n_out=int(out_coords.size(0)),
        bm=int(max_bm),
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=tensors[9],
        sorted=bool(sorted),
    )


def build_center_last_reverse_runtime_from_full_runtime(
    forward_runtime: FinalizedRowTemplateCenterLastRuntime,
    out_coords: torch.Tensor,
    out_spatial: Tuple[int, int, int] | list[int],
    *,
    max_bm: int,
    sorted: bool | None = None,
) -> FinalizedRowTemplateCenterLastRuntime:
    out_spatial = tuple(int(v) for v in out_spatial)
    effective_sorted = bool(forward_runtime.sorted if sorted is None else sorted)
    if (
        forward_runtime.reverse_dense_out_in_map is not None
        and forward_runtime.reverse_masks is not None
        and int(forward_runtime.reverse_dense_out_in_map.size(0)) == int(out_coords.size(0))
    ):
        tensors = _C.gtsparse3d_finalize_row_template_center_last_build_runtime_from_dense_out_in_map(
            forward_runtime.reverse_dense_out_in_map,
            forward_runtime.reverse_masks,
            int(max_bm),
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
            int(out_coords.size(0)),
            int(max_bm),
            bool(effective_sorted),
        )
    return _from_cuda_runtime_tuple(
        tensors,
        n_out=int(out_coords.size(0)),
        bm=int(max_bm),
        out_coords=out_coords,
        out_spatial=out_spatial,
        coord_hashmap=None,
        sorted=bool(effective_sorted),
    )


def classify_center_last_templates_from_dense_out_in_map(
    dense_out_in_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dense_out_in_map.dtype != torch.int32:
        raise TypeError("dense_out_in_map must be int32")
    if dense_out_in_map.dim() != 2 or int(dense_out_in_map.size(1)) != NUM_LOGICAL_OFFSETS:
        raise ValueError("dense_out_in_map must have shape [N_out, 27] in actual flatten order")

    device = dense_out_in_map.device
    match_dense = dense_out_in_map.index_select(1, _match_logical_to_actual_index(device))
    active = match_dense >= 0
    template_ids = torch.full((int(dense_out_in_map.size(0)),), -1, dtype=torch.int32, device=device)
    remaining = torch.ones((int(dense_out_in_map.size(0)),), dtype=torch.bool, device=device)

    for template_id, reject_mask_seq in enumerate(TEMPLATE_REJECT_MASKS):
        reject_mask = torch.tensor(reject_mask_seq, device=device, dtype=torch.bool)
        blocked = active & reject_mask.view(1, NUM_LOGICAL_OFFSETS)
        fits = remaining & (~blocked.any(dim=1))
        template_ids[fits] = int(template_id)
        remaining &= ~fits

    if bool(remaining.any().item()):
        raise RuntimeError("center-last classifier left some rows unclassified")

    template_counts = torch.zeros((NUM_TEMPLATES,), dtype=torch.int32, device=device)
    for template_id in range(NUM_TEMPLATES):
        template_counts[template_id] = int((template_ids == template_id).sum().item())

    return match_dense, template_ids, template_counts


def center_last_fp32_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
    *,
    setting: int | None = None,
) -> torch.Tensor:
    if setting is not None:
        setting = int(setting)
        if setting not in (1, 2, 3):
            raise ValueError("center-last fp32 path currently supports setting in {1, 2, 3}")
    if features.dtype != torch.float32:
        raise TypeError("center-last row-template path requires float32 features")
    if logical_weight.dtype != torch.float32:
        raise TypeError("center-last row-template path requires float32 logical_weight")
    if features.dim() != 2:
        raise ValueError("features must be 2D [N_in, Cin]")
    if logical_weight.dim() != 2:
        raise ValueError("logical_weight must be flattened 2D [27 * Cin, Cout]")
    if not features.is_cuda or not logical_weight.is_cuda:
        raise ValueError("center-last fp32 path currently requires CUDA tensors")
    if not features.is_contiguous():
        raise ValueError("center-last fp32 path requires contiguous features")
    if not logical_weight.is_contiguous():
        raise ValueError("center-last fp32 path requires contiguous logical_weight")
    cin = int(features.size(1))
    if int(logical_weight.size(0)) != NUM_LOGICAL_OFFSETS * cin:
        raise ValueError("logical_weight first dimension must equal 27 * Cin")

    if setting is None:
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp32_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp32_forward
        )
    elif setting == 1:
        cout = int(logical_weight.size(1))
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp32_setting1_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp32_setting1_forward
        )
    elif setting == 2:
        expected_bk = FP32_SETTING_BK[setting]
        expected_bn = FP32_SETTING_BN[setting]
        cout = int(logical_weight.size(1))
        if cin % expected_bk != 0:
            raise ValueError(f"center-last fp32 setting{setting} path currently requires Cin % {expected_bk} == 0")
        if cout % expected_bn != 0:
            raise ValueError(f"center-last fp32 setting{setting} path currently requires Cout % {expected_bn} == 0")
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp32_setting2_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp32_setting2_forward
        )
    else:
        expected_bk = FP32_SETTING_BK[setting]
        expected_bn = FP32_SETTING_BN[setting]
        cout = int(logical_weight.size(1))
        if cin % expected_bk != 0:
            raise ValueError(f"center-last fp32 setting{setting} path currently requires Cin % {expected_bk} == 0")
        if cout % expected_bn != 0:
            raise ValueError(f"center-last fp32 setting{setting} path currently requires Cout % {expected_bn} == 0")
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
        int(runtime.n_out),
    )


def center_last_fp32_setting1_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
) -> torch.Tensor:
    return center_last_fp32_conv(features, logical_weight, runtime, setting=1)


def center_last_fp32_setting2_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
) -> torch.Tensor:
    return center_last_fp32_conv(features, logical_weight, runtime, setting=2)


def center_last_fp32_setting3_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
) -> torch.Tensor:
    return center_last_fp32_conv(features, logical_weight, runtime, setting=3)


def center_last_fp16_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
    *,
    setting: int | None = None,
) -> torch.Tensor:
    if setting is not None:
        setting = int(setting)
        if setting not in (1, 2, 3):
            raise ValueError("center-last fp16 path currently supports setting in {1, 2, 3}")
    if features.dtype != torch.float16:
        raise TypeError("center-last row-template fp16 path requires float16 features")
    if logical_weight.dtype != torch.float16:
        raise TypeError("center-last row-template fp16 path requires float16 logical_weight")
    if features.dim() != 2:
        raise ValueError("features must be 2D [N_in, Cin]")
    if logical_weight.dim() != 2:
        raise ValueError("logical_weight must be flattened 2D [27 * Cin, Cout]")
    if not features.is_cuda or not logical_weight.is_cuda:
        raise ValueError("center-last fp16 path currently requires CUDA tensors")
    if not features.is_contiguous():
        raise ValueError("center-last fp16 path requires contiguous features")
    if not logical_weight.is_contiguous():
        raise ValueError("center-last fp16 path requires contiguous logical_weight")
    cin = int(features.size(1))
    if int(logical_weight.size(0)) != NUM_LOGICAL_OFFSETS * cin:
        raise ValueError("logical_weight first dimension must equal 27 * Cin")

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
        expected_bk = FP16_SETTING_BK[setting]
        expected_bn = FP16_SETTING_BN[setting]
        cout = int(logical_weight.size(1))
        if cin % expected_bk != 0:
            raise ValueError(f"center-last fp16 setting{setting} path currently requires Cin % {expected_bk} == 0")
        if cout % expected_bn != 0:
            raise ValueError(f"center-last fp16 setting{setting} path currently requires Cout % {expected_bn} == 0")
        fn = (
            _C.gtsparse3d_finalize_row_template_center_last_fp16_setting2_sorted_forward
            if runtime.sorted
            else _C.gtsparse3d_finalize_row_template_center_last_fp16_setting2_forward
        )
    else:
        expected_bk = FP16_SETTING_BK[setting]
        expected_bn = FP16_SETTING_BN[setting]
        cout = int(logical_weight.size(1))
        if cin % expected_bk != 0:
            raise ValueError(f"center-last fp16 setting{setting} path currently requires Cin % {expected_bk} == 0")
        if cout % expected_bn != 0:
            raise ValueError(f"center-last fp16 setting{setting} path currently requires Cout % {expected_bn} == 0")
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
        int(runtime.n_out),
    )


def center_last_fp16_setting1_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
) -> torch.Tensor:
    return center_last_fp16_conv(features, logical_weight, runtime, setting=1)


def center_last_fp16_setting2_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
) -> torch.Tensor:
    return center_last_fp16_conv(features, logical_weight, runtime, setting=2)


def center_last_fp16_setting3_conv(
    features: torch.Tensor,
    logical_weight: torch.Tensor,
    runtime: FinalizedRowTemplateCenterLastRuntime,
) -> torch.Tensor:
    return center_last_fp16_conv(features, logical_weight, runtime, setting=3)
