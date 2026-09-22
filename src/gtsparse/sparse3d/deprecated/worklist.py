"""Worklist utilities for GTSparse sparse tensors.

The worklist (page_ids, local_linears) tells the GEMM kernel where each
feature row lives spatially, enabling index-grid-based neighbor lookup.

For SubMConv3d: worklist is in INPUT order (row i → worklist[i]).
Output features are written in the same order → index grid stays valid.

For full convolution: worklist defines OUTPUT positions and order.
A new index grid is built for the output.
"""

from typing import Tuple

import torch

from .index_grid import IndexGridMetadata, _HAS_CUDA_EXT

try:
    from gtsparse import _C
except ImportError:
    _C = None


def _page_local_to_coords(
    active_subgrid_coords: torch.Tensor,
    page_ids: torch.Tensor,
    local_linear_indices: torch.Tensor,
    subgrid_shape,
) -> torch.Tensor:
    """Convert (page_id, local_linear_index) pairs to (batch, d, h, w) coords."""
    subgrid = [int(v) for v in subgrid_shape]
    page_coords = active_subgrid_coords[page_ids]  # int32 indexing
    local = local_linear_indices.int()  # int16 → int32 for arithmetic

    sh, sw = subgrid[1], subgrid[2]
    local_d = local // (sh * sw)
    local_h = (local % (sh * sw)) // sw
    local_w = local % sw

    coords = torch.empty((page_ids.numel(), 4), dtype=torch.int32, device=page_ids.device)
    coords[:, 0] = page_coords[:, 0]
    coords[:, 1] = page_coords[:, 1] * subgrid[0] + local_d
    coords[:, 2] = page_coords[:, 2] * sh + local_h
    coords[:, 3] = page_coords[:, 3] * sw + local_w
    return coords


def build_subm_worklist_reference(
    index_grid_meta: IndexGridMetadata,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build SubM worklist in scan order via Python/Torch ops.

    The worklist defines the GEMM kernel's traversal order (scan order for
    cache locality). Output write position is determined by index_grid lookup,
    not by worklist position.

    Returns:
        (page_ids [N] int32, local_linears [N] int16) in scan order.
    """
    leaf_grid = index_grid_meta.tensors["leaf_grid"]
    flat_leaf = leaf_grid.view(leaf_grid.size(0), -1)
    active_positions = (flat_leaf >= 0).nonzero(as_tuple=False)

    if active_positions.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int32, device=device),
            torch.empty((0,), dtype=torch.int16, device=device),
        )

    page_ids = active_positions[:, 0].int()
    local_linears = active_positions[:, 1].to(torch.int16)
    return page_ids, local_linears


def build_subm_worklist_cuda(
    index_grid_meta: IndexGridMetadata,
    expected_count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build SubM worklist in scan order via CUDA kernel."""
    active_subgrid_coords = index_grid_meta.tensors["active_subgrid_coords"]
    leaf_grid = index_grid_meta.tensors["leaf_grid"]

    page_ids, local_linears = _C.gtsparse3d_build_subm_worklist(
        active_subgrid_coords, leaf_grid, expected_count,
    )
    return page_ids, local_linears


def build_subm_worklist(
    index_grid_meta: IndexGridMetadata,
    expected_count: int,
    device: torch.device,
    *,
    backend: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build SubM worklist in scan order.

    Returns (page_ids [N] int32, local_linears [N] int16).
    """
    if backend == "python":
        return build_subm_worklist_reference(index_grid_meta, device)
    if backend == "cuda":
        return build_subm_worklist_cuda(index_grid_meta, expected_count)
    if device.type == "cuda" and _HAS_CUDA_EXT:
        return build_subm_worklist_cuda(index_grid_meta, expected_count)
    return build_subm_worklist_reference(index_grid_meta, device)
