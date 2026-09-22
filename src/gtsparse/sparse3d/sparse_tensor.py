"""Minimal SparseConvTensor-compatible container for sparse runtimes."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


_UNSET = object()


@dataclass
class IndiceData:
    """Stores index mapping between paired SparseConv3d / SparseInverseConv3d."""

    in_indices: torch.Tensor
    out_indices: torch.Tensor
    in_spatial_shape: List[int]
    out_spatial_shape: List[int]
    kernel_size: Optional[Tuple[int, int, int]] = None
    stride: Optional[Tuple[int, int, int]] = None
    padding: Optional[Tuple[int, int, int]] = None
    dilation: Optional[Tuple[int, int, int]] = None


class GTSparseSparseConvTensor:
    """Sparse tensor wrapper used by the mainline expanded-worklist kernels."""

    def __init__(
        self,
        features: torch.Tensor,
        indices: torch.Tensor,
        spatial_shape,
        batch_size: int,
        *,
        index_grid: Optional[torch.Tensor] = None,
        coord_hashmap: Optional[torch.Tensor] = None,
        metadata: object | None = None,
    ):
        self.features = features
        self._coords = indices
        self.spatial_shape = spatial_shape if isinstance(spatial_shape, list) else list(spatial_shape)
        self.batch_size = batch_size
        self.indice_dict: Dict[str, IndiceData] = {}
        self._runtime_cache: dict[str, object] = {}
        self._index_grid: Optional[torch.Tensor] = index_grid
        self._coord_hashmap: Optional[torch.Tensor] = coord_hashmap
        self.metadata: object | None = metadata

    @classmethod
    def _from_components(
        cls,
        *,
        features: torch.Tensor,
        coords: torch.Tensor,
        spatial_shape,
        batch_size: int,
        indice_dict: Dict[str, IndiceData] | None,
        runtime_cache: dict[str, object] | None,
        index_grid: Optional[torch.Tensor],
        coord_hashmap: Optional[torch.Tensor],
        metadata: object | None,
    ) -> "GTSparseSparseConvTensor":
        out = object.__new__(cls)
        out.features = features
        out._coords = coords
        out.spatial_shape = spatial_shape if isinstance(spatial_shape, list) else list(spatial_shape)
        out.batch_size = int(batch_size)
        out.indice_dict = {} if indice_dict is None else indice_dict
        out._runtime_cache = {} if runtime_cache is None else runtime_cache
        out._index_grid = index_grid
        out._coord_hashmap = coord_hashmap
        out.metadata = metadata
        return out

    @property
    def device(self):
        return self.features.device

    @property
    def indices(self) -> torch.Tensor:
        return self._coords

    @property
    def index_grid(self) -> torch.Tensor:
        if self._index_grid is None:
            self._build_index_grid()
        return self._index_grid

    @property
    def coord_hashmap(self) -> Optional[torch.Tensor]:
        return self._coord_hashmap

    def _build_index_grid(self):
        D, H, W = self.spatial_shape
        grid = torch.full(
            (self.batch_size, D, H, W), -1, dtype=torch.int32, device=self.device
        )
        idx = self.indices
        linear_idx = torch.arange(len(idx), dtype=torch.int32, device=self.device)
        grid[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = linear_idx
        self._index_grid = grid

    def invalidate_index_grid(self):
        self._index_grid = None

    def active_mask(self) -> torch.Tensor:
        if self._index_grid is not None:
            return self._index_grid >= 0
        mask = torch.zeros(
            self.batch_size, *self.spatial_shape, dtype=torch.bool, device=self.device
        )
        idx = self.indices
        mask[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = True
        return mask

    def dense(self, channels_first: bool = True) -> torch.Tensor:
        D, H, W = self.spatial_shape
        out = torch.zeros(
            self.batch_size,
            D,
            H,
            W,
            self.features.shape[1],
            dtype=self.features.dtype,
            device=self.device,
        )
        idx = self.indices
        out[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]] = self.features
        if channels_first:
            out = out.permute(0, 4, 1, 2, 3)
        return out

    def replace_feature(
        self,
        new_features: torch.Tensor,
        *,
        metadata: object | None | object = _UNSET,
        coord_hashmap: Optional[torch.Tensor] | object = _UNSET,
        index_grid: Optional[torch.Tensor] | object = _UNSET,
    ) -> "GTSparseSparseConvTensor":
        return GTSparseSparseConvTensor._from_components(
            features=new_features,
            coords=self._coords,
            spatial_shape=self.spatial_shape,
            batch_size=self.batch_size,
            indice_dict=self.indice_dict,
            runtime_cache=self._runtime_cache,
            index_grid=self._index_grid if index_grid is _UNSET else index_grid,
            coord_hashmap=self._coord_hashmap if coord_hashmap is _UNSET else coord_hashmap,
            metadata=self.metadata if metadata is _UNSET else metadata,
        )

    def replace_feature_(self, new_features: torch.Tensor) -> "GTSparseSparseConvTensor":
        self.features = new_features
        return self

    def replace_sparse(
        self,
        *,
        new_features: torch.Tensor,
        new_coords: torch.Tensor,
        new_spatial_shape,
        new_batch_size: int,
        metadata: object | None,
        coord_hashmap: Optional[torch.Tensor],
        index_grid: Optional[torch.Tensor] = None,
    ) -> "GTSparseSparseConvTensor":
        return GTSparseSparseConvTensor._from_components(
            features=new_features,
            coords=new_coords,
            spatial_shape=new_spatial_shape,
            batch_size=new_batch_size,
            indice_dict=self.indice_dict,
            runtime_cache=self._runtime_cache,
            index_grid=index_grid,
            coord_hashmap=coord_hashmap,
            metadata=metadata,
        )

    @staticmethod
    def from_dense(dense: torch.Tensor) -> "GTSparseSparseConvTensor":
        B, C, D, H, W = dense.shape
        dense_nhwc = dense.permute(0, 2, 3, 4, 1)
        mask = dense_nhwc.abs().sum(dim=-1) > 0
        indices = mask.nonzero(as_tuple=False).int()
        features = dense_nhwc[mask]
        return GTSparseSparseConvTensor(features, indices, [D, H, W], B)
