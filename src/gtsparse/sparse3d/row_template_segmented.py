from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from gtsparse import _C
from gtsparse.sparse3d.row_template_segmented_family_layout import (
    FAMILY_CENTER,
    FAMILY_FULL27_END,
    FAMILY_FULL27,
    FAMILY_SINGLE,
    FAMILY_SKIP1EVERY3_END,
    FAMILY_SKIP1EVERY3_START,
    FAMILY_SKIP1EVERY3,
    FAMILY_SKIP2EVERY3_END,
    FAMILY_SKIP2EVERY3_START,
    FAMILY_SKIP2EVERY3,
    FAMILY_SINGLE_END,
    SEGMENT_PAYLOAD_WIDTHS,
    SEGMENT_TEMPLATE_COUNTS,
    TEMPLATE_LOCAL_INDEX,
    family_from_template_id,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

_CENTER_OFFSET = 13
_RING_ORDER = (
    0, 9, 18, 3, 12, 21, 6, 15, 24, 1, 10, 19, 4,
    22, 7, 16, 25, 2, 11, 20, 5, 14, 23, 8, 17, 26,
)


@dataclass(slots=True)
class SegmentedSubmCodebook:
    codebook_w1: torch.Tensor
    codebook_w2: torch.Tensor
    codebook_w9: torch.Tensor
    codebook_w18: torch.Tensor
    codebook_w27: torch.Tensor
    template_counts: torch.Tensor
    template_ids: torch.Tensor
    n_out: int

    @property
    def template_stride(self) -> int:
        return int(self.codebook_w1.size(1))

    @property
    def segments(self) -> tuple[torch.Tensor, ...]:
        return (
            self.codebook_w1,
            self.codebook_w2,
            self.codebook_w9,
            self.codebook_w18,
            self.codebook_w27,
        )


@dataclass(slots=True)
class SegmentedCompactTilelistRuntime:
    codebook_w1: torch.Tensor
    codebook_w2: torch.Tensor
    codebook_w9: torch.Tensor
    codebook_w18: torch.Tensor
    codebook_w27: torch.Tensor
    tile_family_ids: torch.Tensor
    tile_template_ids: torch.Tensor
    tile_storage_row_offsets: torch.Tensor
    n_out: int
    bm: int

    @property
    def num_tiles(self) -> int:
        return int(self.tile_template_ids.numel())

    @property
    def segments(self) -> tuple[torch.Tensor, ...]:
        return (
            self.codebook_w1,
            self.codebook_w2,
            self.codebook_w9,
            self.codebook_w18,
            self.codebook_w27,
        )


@dataclass(slots=True)
class SegmentedFlatTiledRuntime:
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


def _template_offsets(template_id: int) -> list[int]:
    template_id = int(template_id)
    local_idx = int(TEMPLATE_LOCAL_INDEX[template_id])
    if template_id == 0:
        return [_CENTER_OFFSET]
    if template_id <= FAMILY_SINGLE_END:
        return [_CENTER_OFFSET, _RING_ORDER[local_idx]]
    if template_id <= FAMILY_SKIP2EVERY3_END:
        start = template_id - FAMILY_SKIP2EVERY3_START
        return [_CENTER_OFFSET] + [
            _RING_ORDER[(start + rel_pos) % 26]
            for rel_pos in range(26)
            if (rel_pos % 3) >= 2
        ]
    if template_id <= FAMILY_SKIP1EVERY3_END:
        start = template_id - FAMILY_SKIP1EVERY3_START
        return [_CENTER_OFFSET] + [
            _RING_ORDER[(start + rel_pos) % 26]
            for rel_pos in range(26)
            if (rel_pos % 3) != 0
        ]
    if template_id <= FAMILY_FULL27_END:
        return list(range(27))
    raise ValueError(f"unexpected template_id={template_id}")


_TEMPLATE_OFFSETS = tuple(
    _template_offsets(tid) for tid in range(sum(int(v) for v in SEGMENT_TEMPLATE_COUNTS))
)
_TEMPLATE_MASKS = tuple(
    sum(1 << off for off in offsets)
    for offsets in _TEMPLATE_OFFSETS
)


def merge_segmented_padding_tails(
    runtime: SegmentedSubmCodebook,
    out_in_map: torch.Tensor,
    *,
    bm: int,
) -> SegmentedSubmCodebook:
    if out_in_map.dtype != torch.int32:
        raise TypeError("out_in_map must be int32")
    if out_in_map.dim() != 2 or int(out_in_map.size(1)) != 27:
        raise ValueError("out_in_map must be [N_out, 27]")
    if int(out_in_map.size(0)) != int(runtime.n_out):
        raise ValueError("out_in_map row count must match runtime.n_out")
    if int(bm) <= 0:
        raise ValueError("bm must be positive")

    device = out_in_map.device
    out_in_map_cpu = out_in_map.detach().cpu().to(torch.int32)
    template_ids_cpu = runtime.template_ids.detach().cpu().to(torch.int64)
    n_out = int(runtime.n_out)
    num_templates = len(_TEMPLATE_OFFSETS)

    row_masks: list[int] = []
    for row_idx in range(n_out):
        active_offsets = torch.nonzero(out_in_map_cpu[row_idx] >= 0, as_tuple=False).view(-1).tolist()
        row_mask = 0
        for off in active_offsets:
            row_mask |= 1 << int(off)
        row_masks.append(row_mask)

    rows_by_template: list[list[int]] = [[] for _ in range(num_templates)]
    for out_row, template_id in enumerate(template_ids_cpu.tolist()):
        rows_by_template[int(template_id)].append(int(out_row))

    for template_id in range(num_templates - 1):
        rows = rows_by_template[template_id]
        keep = (len(rows) // int(bm)) * int(bm)
        if keep == len(rows):
            continue
        tail_rows = rows[keep:]
        rows_by_template[template_id] = rows[:keep]
        for out_row in tail_rows:
            row_mask = row_masks[out_row]
            moved = False
            for dst_template_id in range(template_id + 1, num_templates):
                if (row_mask & ~_TEMPLATE_MASKS[dst_template_id]) == 0:
                    rows_by_template[dst_template_id].append(out_row)
                    moved = True
                    break
            if not moved:
                rows_by_template[template_id].append(out_row)

    segment_shapes = [
        (SEGMENT_TEMPLATE_COUNTS[i], runtime.template_stride, SEGMENT_PAYLOAD_WIDTHS[i])
        for i in range(len(SEGMENT_TEMPLATE_COUNTS))
    ]
    segments = [
        torch.full(shape, -1, dtype=torch.int32)
        for shape in segment_shapes
    ]
    template_counts = torch.zeros((num_templates,), dtype=torch.int32)
    template_ids = torch.full((n_out,), -1, dtype=torch.int32)

    family_starts = []
    running = 0
    for count in SEGMENT_TEMPLATE_COUNTS:
        family_starts.append(running)
        running += int(count)

    for template_id, rows in enumerate(rows_by_template):
        count = len(rows)
        if count <= 0:
            continue
        template_counts[template_id] = int(count)
        template_ids[torch.tensor(rows, dtype=torch.long)] = int(template_id)
        family_idx = 0
        while family_idx + 1 < len(family_starts) and template_id >= family_starts[family_idx + 1]:
            family_idx += 1
        local_idx = template_id - family_starts[family_idx]
        rows_tensor = torch.tensor(rows, dtype=torch.long)
        offsets = _TEMPLATE_OFFSETS[template_id]
        payload = out_in_map_cpu.index_select(0, rows_tensor)[:, offsets]
        segments[family_idx][local_idx, :count, 0] = rows_tensor.to(torch.int32)
        segments[family_idx][local_idx, :count, 1 : 1 + len(offsets)] = payload

    segments = [seg.to(device=device) for seg in segments]
    return SegmentedSubmCodebook(
        codebook_w1=segments[0],
        codebook_w2=segments[1],
        codebook_w9=segments[2],
        codebook_w18=segments[3],
        codebook_w27=segments[4],
        template_counts=template_counts.to(device=device),
        template_ids=template_ids.to(device=device),
        n_out=n_out,
    )


def build_segmented_subm_codebook_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
) -> SegmentedSubmCodebook:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("segmented subm prototype currently requires kernel_size=(3, 3, 3)")
    if tuple(int(v) for v in padding) != (1, 1, 1):
        raise ValueError("segmented subm prototype currently requires padding=(1, 1, 1)")
    if tuple(int(v) for v in dilation) != (1, 1, 1):
        raise ValueError("segmented subm prototype currently requires dilation=(1, 1, 1)")
    (
        codebook_w1,
        codebook_w2,
        codebook_w9,
        codebook_w18,
        codebook_w27,
        template_counts,
        template_ids,
    ) = _C.gtsparse3d_segmented_build_subm_codebook_from_coords(
        st.indices.contiguous(),
        int(max_bm),
    )
    segments = (codebook_w1, codebook_w2, codebook_w9, codebook_w18, codebook_w27)
    template_stride = int(codebook_w1.size(1))
    for idx, (segment, expected_templates, expected_width) in enumerate(
        zip(segments, SEGMENT_TEMPLATE_COUNTS, SEGMENT_PAYLOAD_WIDTHS, strict=True)
    ):
        if segment.dtype != torch.int32:
            raise TypeError(f"segment {idx} must be int32")
        if segment.dim() != 3:
            raise ValueError(f"segment {idx} must be rank-3")
        if int(segment.size(0)) != int(expected_templates):
            raise ValueError(f"segment {idx} template dim mismatch")
        if int(segment.size(1)) != template_stride:
            raise ValueError(f"segment {idx} template_stride mismatch")
        if int(segment.size(2)) != int(expected_width):
            raise ValueError(f"segment {idx} width mismatch")
    return SegmentedSubmCodebook(
        codebook_w1=codebook_w1,
        codebook_w2=codebook_w2,
        codebook_w9=codebook_w9,
        codebook_w18=codebook_w18,
        codebook_w27=codebook_w27,
        template_counts=template_counts,
        template_ids=template_ids,
        n_out=int(st.indices.size(0)),
    )


def build_segmented_flat_tiled_runtime_from_coords(
    st: GTSparseSparseConvTensor,
    kernel_size: Tuple[int, int, int],
    padding: Tuple[int, int, int],
    dilation: Tuple[int, int, int] = (1, 1, 1),
    *,
    max_bm: int,
) -> SegmentedFlatTiledRuntime:
    if tuple(int(v) for v in kernel_size) != (3, 3, 3):
        raise ValueError("segmented flat tiled prototype currently requires kernel_size=(3, 3, 3)")
    if tuple(int(v) for v in padding) != (1, 1, 1):
        raise ValueError("segmented flat tiled prototype currently requires padding=(1, 1, 1)")
    if tuple(int(v) for v in dilation) != (1, 1, 1):
        raise ValueError("segmented flat tiled prototype currently requires dilation=(1, 1, 1)")
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
    ) = _C.gtsparse3d_segmented_build_flat_tiled_runtime_from_coords(
        st.indices.contiguous(),
        int(max_bm),
    )
    return SegmentedFlatTiledRuntime(
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
    )


def build_segmented_compact_tilelist_runtime(
    runtime: SegmentedSubmCodebook,
    *,
    bm: int,
) -> SegmentedCompactTilelistRuntime:
    if int(bm) <= 0:
        raise ValueError("bm must be positive")
    device = runtime.codebook_w1.device
    family_buffers = [[] for _ in range(len(SEGMENT_TEMPLATE_COUNTS))]
    family_row_totals = [0 for _ in range(len(SEGMENT_TEMPLATE_COUNTS))]
    tile_family_ids: list[int] = []
    tile_template_ids: list[int] = []
    tile_storage_row_offsets: list[int] = []
    counts_cpu = runtime.template_counts.detach().cpu().tolist()

    for template_id, count_val in enumerate(counts_cpu):
        count = int(count_val)
        if count <= 0:
            continue
        family_id = int(family_from_template_id(template_id))
        local_idx = int(TEMPLATE_LOCAL_INDEX[template_id])
        src = runtime.segments[family_id][local_idx, :count]
        padded = ((count + int(bm) - 1) // int(bm)) * int(bm)
        row_offset = int(family_row_totals[family_id])
        family_buffers[family_id].append((row_offset, src))
        family_row_totals[family_id] += padded
        for tile_row in range(0, padded, int(bm)):
            tile_family_ids.append(family_id)
            tile_template_ids.append(int(template_id))
            tile_storage_row_offsets.append(row_offset + tile_row)

    segments: list[torch.Tensor] = []
    for family_id, (rows, width) in enumerate(zip(family_row_totals, SEGMENT_PAYLOAD_WIDTHS, strict=True)):
        seg = torch.full(
            (int(rows), int(width)),
            -1,
            device=device,
            dtype=torch.int32,
        )
        for row_offset, src in family_buffers[family_id]:
            count = int(src.size(0))
            seg[row_offset : row_offset + count] = src
        segments.append(seg)

    return SegmentedCompactTilelistRuntime(
        codebook_w1=segments[0],
        codebook_w2=segments[1],
        codebook_w9=segments[2],
        codebook_w18=segments[3],
        codebook_w27=segments[4],
        tile_family_ids=torch.tensor(tile_family_ids, device=device, dtype=torch.int32),
        tile_template_ids=torch.tensor(tile_template_ids, device=device, dtype=torch.int32),
        tile_storage_row_offsets=torch.tensor(tile_storage_row_offsets, device=device, dtype=torch.int32),
        n_out=int(runtime.n_out),
        bm=int(bm),
    )


def build_segmented_flat_tiled_runtime(
    runtime: SegmentedSubmCodebook,
    *,
    bm: int,
) -> SegmentedFlatTiledRuntime:
    if int(bm) <= 0:
        raise ValueError("bm must be positive")
    device = runtime.codebook_w1.device
    slot_widths = tuple(int(width) - 1 for width in SEGMENT_PAYLOAD_WIDTHS)
    out_rows_chunks: list[torch.Tensor] = []
    template_ids_chunks: list[torch.Tensor] = []
    input_row_offsets_chunks: list[torch.Tensor] = []
    counts_cpu = runtime.template_counts.detach().cpu().tolist()
    template_stride = int(runtime.template_stride)
    input_segments = [
        torch.full(
            (int(template_count), template_stride, int(slot_width)),
            -1,
            device=device,
            dtype=torch.int32,
        )
        for template_count, slot_width in zip(SEGMENT_TEMPLATE_COUNTS, slot_widths, strict=True)
    ]

    for template_id, count_val in enumerate(counts_cpu):
        count = int(count_val)
        if count <= 0:
            continue
        family_id = int(family_from_template_id(template_id))
        local_idx = int(TEMPLATE_LOCAL_INDEX[template_id])
        src = runtime.segments[family_id][local_idx, :count]
        padded = ((count + int(bm) - 1) // int(bm)) * int(bm)
        slot_width = slot_widths[family_id]
        out_rows = torch.full((padded,), -1, device=device, dtype=torch.int32)
        out_rows[:count] = src[:, 0]
        out_rows_chunks.append(out_rows)
        template_ids_chunks.append(
            torch.full((padded,), int(template_id), device=device, dtype=torch.int32)
        )
        input_row_offsets_chunks.append(
            torch.arange(padded, device=device, dtype=torch.int32)
        )
        input_segments[family_id][local_idx, :count] = src[:, 1 : 1 + slot_width]

    if out_rows_chunks:
        out_rows = torch.cat(out_rows_chunks, dim=0)
        template_ids = torch.cat(template_ids_chunks, dim=0)
        input_row_offsets = torch.cat(input_row_offsets_chunks, dim=0)
    else:
        out_rows = torch.empty((0,), device=device, dtype=torch.int32)
        template_ids = torch.empty((0,), device=device, dtype=torch.int32)
        input_row_offsets = torch.empty((0,), device=device, dtype=torch.int32)
    padded_row_count = torch.tensor([int(out_rows.numel())], device=device, dtype=torch.int32)

    return SegmentedFlatTiledRuntime(
        out_rows=out_rows,
        input_rows_w1=input_segments[0],
        input_rows_w2=input_segments[1],
        input_rows_w9=input_segments[2],
        input_rows_w18=input_segments[3],
        input_rows_w27=input_segments[4],
        template_ids=template_ids,
        input_row_offsets=input_row_offsets,
        padded_row_count=padded_row_count,
        n_out=int(runtime.n_out),
        bm=int(bm),
    )


def row_template_segmented_subm_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: SegmentedSubmCodebook,
    *,
    config_id: int = 0,
    tail_mode: str = "ticket",
) -> torch.Tensor:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("segmented igemm kernel requires float16 features and weight")
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
    tail_mode_id = 0 if tail_mode == "ticket" else 1
    return _C.gtsparse3d_row_template_segmented_direct_forward(
        features,
        w,
        runtime.codebook_w1.contiguous(),
        runtime.codebook_w2.contiguous(),
        runtime.codebook_w9.contiguous(),
        runtime.codebook_w18.contiguous(),
        runtime.codebook_w27.contiguous(),
        runtime.template_counts.contiguous(),
        int(runtime.n_out),
        int(config_id),
        tail_mode_id,
    )


def row_template_segmented_compact_tilelist_subm_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: SegmentedCompactTilelistRuntime,
    *,
    config_id: int = 1,
) -> torch.Tensor:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("compact tilelist igemm kernel requires float16 features and weight")
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
    return _C.gtsparse3d_row_template_segmented_compact_tilelist_forward(
        features,
        w,
        runtime.codebook_w1.contiguous(),
        runtime.codebook_w2.contiguous(),
        runtime.codebook_w9.contiguous(),
        runtime.codebook_w18.contiguous(),
        runtime.codebook_w27.contiguous(),
        runtime.tile_family_ids.contiguous(),
        runtime.tile_template_ids.contiguous(),
        runtime.tile_storage_row_offsets.contiguous(),
        int(runtime.n_out),
        int(config_id),
    )


def row_template_segmented_flat_tiled_subm_conv3d(
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime: SegmentedFlatTiledRuntime,
    *,
    config_id: int = 1,
) -> torch.Tensor:
    if features.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("flat tiled igemm kernel requires float16 features and weight")
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
    return _C.gtsparse3d_row_template_segmented_flat_tiled_forward(
        features,
        w,
        runtime.out_rows.contiguous(),
        runtime.input_rows_w1.contiguous(),
        runtime.input_rows_w2.contiguous(),
        runtime.input_rows_w9.contiguous(),
        runtime.input_rows_w18.contiguous(),
        runtime.input_rows_w27.contiguous(),
        runtime.template_ids.contiguous(),
        runtime.input_row_offsets.contiguous(),
        runtime.padded_row_count.contiguous(),
        int(runtime.n_out),
        int(runtime.bm),
        int(config_id),
    )
