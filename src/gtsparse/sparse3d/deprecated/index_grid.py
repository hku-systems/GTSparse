"""Hierarchical index-grid builders for GTSparse sparse tensors.

CUDA is the preferred backend when available, while the Python/Torch version
remains as a readable fallback and a reference for tests.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import torch

try:
    from gtsparse import _C
    _HAS_CUDA_EXT = hasattr(_C, "gtsparse3d_build_hierarchical_index_grid")
except ImportError:
    _C = None
    _HAS_CUDA_EXT = False


@dataclass
class IndexGridMetadata:
    layout: str
    spatial_shape: List[int]
    batch_size: int
    grid_shape: List[int]
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    attrs: Dict[str, Any] = field(default_factory=dict)

    def validate(self, spatial_shape, batch_size: int) -> None:
        assert list(self.spatial_shape) == list(spatial_shape), \
            "index_grid_meta.spatial_shape does not match sparse tensor spatial_shape"
        assert self.batch_size == batch_size, \
            "index_grid_meta.batch_size does not match sparse tensor batch_size"


def _normalize_subgrid_shape(subgrid_shape: Sequence[int] | int) -> List[int]:
    if isinstance(subgrid_shape, int):
        return [subgrid_shape] * 3
    values = list(subgrid_shape)
    assert len(values) == 3 and all(v > 0 for v in values), \
        "subgrid_shape must be a positive int or length-3 sequence"
    return values


def _build_hierarchical_index_grid_reference(
    indices: torch.Tensor,
    spatial_shape,
    batch_size: int,
    subgrid_shape: Sequence[int] | int = (8, 8, 8),
) -> IndexGridMetadata:
    """Build a two-pass reference hierarchical index grid with Torch ops."""
    assert indices.dim() == 2 and indices.size(1) == 4, "indices must be [N, 4] int32"
    assert indices.dtype == torch.int32, "indices must be int32"
    spatial = list(spatial_shape)
    subgrid = _normalize_subgrid_shape(subgrid_shape)
    device = indices.device

    grid_d = (spatial[0] + subgrid[0] - 1) // subgrid[0]
    grid_h = (spatial[1] + subgrid[1] - 1) // subgrid[1]
    grid_w = (spatial[2] + subgrid[2] - 1) // subgrid[2]
    total_subgrids = batch_size * grid_d * grid_h * grid_w

    if indices.numel() == 0:
        return IndexGridMetadata(
            layout="hierarchical", spatial_shape=spatial, batch_size=batch_size,
            grid_shape=[batch_size, grid_d, grid_h, grid_w],
            tensors={
                "page_grid": torch.full((batch_size, grid_d, grid_h, grid_w), -1, dtype=torch.int32, device=device),
                "subgrid_offsets": torch.full((batch_size, grid_d, grid_h, grid_w), -1, dtype=torch.int32, device=device),
                "leaf_grid": torch.full((0, *subgrid), -1, dtype=torch.int32, device=device),
                "active_subgrid_coords": torch.empty((0, 4), dtype=torch.int32, device=device),
            },
            attrs={"subgrid_shape": subgrid, "num_pages": 0, "levels": 2, "algorithm": "two_pass_reference"},
        )

    # Use int32 arithmetic throughout — indices are already int32
    idx = indices
    batches = idx[:, 0]
    xyz = idx[:, 1:]
    subgrid_t = torch.tensor(subgrid, device=device, dtype=torch.int32)
    subgrid_coords = torch.div(xyz, subgrid_t, rounding_mode="floor")
    local_coords = xyz - subgrid_coords * subgrid_t

    flat_subgrid = (
        ((batches * grid_d + subgrid_coords[:, 0]) * grid_h + subgrid_coords[:, 1]) * grid_w
        + subgrid_coords[:, 2]
    ).long()

    occupancy = torch.zeros(total_subgrids, dtype=torch.bool, device=device)
    occupancy[flat_subgrid] = True

    prefix = occupancy.to(dtype=torch.int32).cumsum(0, dtype=torch.int32) - 1
    page_ids_flat = torch.full((total_subgrids,), -1, dtype=torch.int32, device=device)
    page_ids_flat[occupancy] = prefix[occupancy]
    num_pages = int(occupancy.sum().item())

    subgrid_offsets = page_ids_flat.view(batch_size, grid_d, grid_h, grid_w)

    active_flat = occupancy.nonzero(as_tuple=False).flatten().long()
    if num_pages > 0:
        dhw = grid_d * grid_h * grid_w
        hw = grid_h * grid_w
        active_subgrid_coords = torch.stack([
            active_flat // dhw,
            (active_flat % dhw) // hw,
            (active_flat % hw) // grid_w,
            active_flat % grid_w,
        ], dim=1).to(torch.int32)
    else:
        active_subgrid_coords = torch.empty((0, 4), dtype=torch.int32, device=device)

    leaf_grid = torch.full((num_pages, *subgrid), -1, dtype=torch.int32, device=device)
    if num_pages > 0:
        local_linear = (
            (local_coords[:, 0] * subgrid[1] + local_coords[:, 1]) * subgrid[2]
            + local_coords[:, 2]
        ).long()
        leaf_flat = leaf_grid.view(num_pages, -1)
        feature_rows = torch.arange(indices.size(0), dtype=torch.int32, device=device)
        page_ids = page_ids_flat[flat_subgrid].long()
        leaf_flat[page_ids, local_linear] = feature_rows

    return IndexGridMetadata(
        layout="hierarchical", spatial_shape=spatial, batch_size=batch_size,
        grid_shape=[batch_size, grid_d, grid_h, grid_w],
        tensors={
            "page_grid": subgrid_offsets, "subgrid_offsets": subgrid_offsets,
            "leaf_grid": leaf_grid, "active_subgrid_coords": active_subgrid_coords,
        },
        attrs={"subgrid_shape": subgrid, "num_pages": num_pages, "levels": 2, "algorithm": "two_pass_reference"},
    )


def _build_hierarchical_index_grid_cuda(
    indices: torch.Tensor,
    spatial_shape,
    batch_size: int,
    subgrid_shape: Sequence[int] | int = (8, 8, 8),
) -> IndexGridMetadata:
    """Build a two-pass hierarchical index grid with CUDA kernels."""
    assert indices.dim() == 2 and indices.size(1) == 4, "indices must be [N, 4] int32"
    assert indices.dtype == torch.int32, "indices must be int32"
    assert indices.is_cuda, "CUDA builder requires CUDA indices"
    assert indices.is_contiguous(), "indices must be contiguous"
    spatial = list(spatial_shape)
    subgrid = _normalize_subgrid_shape(subgrid_shape)

    subgrid_offsets, leaf_grid, active_subgrid_coords = _C.gtsparse3d_build_hierarchical_index_grid(
        indices, batch_size, spatial[0], spatial[1], spatial[2],
        subgrid[0], subgrid[1], subgrid[2],
    )

    grid_d = (spatial[0] + subgrid[0] - 1) // subgrid[0]
    grid_h = (spatial[1] + subgrid[1] - 1) // subgrid[1]
    grid_w = (spatial[2] + subgrid[2] - 1) // subgrid[2]

    return IndexGridMetadata(
        layout="hierarchical", spatial_shape=spatial, batch_size=batch_size,
        grid_shape=[batch_size, grid_d, grid_h, grid_w],
        tensors={
            "page_grid": subgrid_offsets, "subgrid_offsets": subgrid_offsets,
            "leaf_grid": leaf_grid, "active_subgrid_coords": active_subgrid_coords,
        },
        attrs={
            "subgrid_shape": subgrid, "num_pages": int(active_subgrid_coords.size(0)),
            "levels": 2, "algorithm": "two_pass_cuda",
        },
    )


def build_hierarchical_index_grid(
    indices: torch.Tensor,
    spatial_shape,
    batch_size: int,
    subgrid_shape: Sequence[int] | int = (8, 8, 8),
    *,
    backend: str = "auto",
) -> IndexGridMetadata:
    if backend == "python":
        return _build_hierarchical_index_grid_reference(indices, spatial_shape, batch_size, subgrid_shape)
    if backend == "cuda":
        return _build_hierarchical_index_grid_cuda(indices, spatial_shape, batch_size, subgrid_shape)
    if indices.is_cuda and _HAS_CUDA_EXT:
        return _build_hierarchical_index_grid_cuda(indices, spatial_shape, batch_size, subgrid_shape)
    return _build_hierarchical_index_grid_reference(indices, spatial_shape, batch_size, subgrid_shape)
