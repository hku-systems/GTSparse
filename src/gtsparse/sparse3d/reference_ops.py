"""Reference sparse 3D operators built on top of native PyTorch kernels."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .sparse_tensor import GTSparseSparseConvTensor, IndiceData


def _normalize_3tuple(value) -> Tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    values = tuple(value)
    if len(values) != 3:
        raise ValueError("expected an int or length-3 sequence")
    return values


def gather_dense_features_at_coords(dense: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Gather `[N, C]` features from dense `[B, C, D, H, W]` at `[N, 4]` coords."""
    dense_nhwc = dense.permute(0, 2, 3, 4, 1)
    idx = coords.long()
    return dense_nhwc[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]


def compare_sparse_outputs(
    actual: GTSparseSparseConvTensor,
    expected: GTSparseSparseConvTensor,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
):
    """Compare sparse outputs after canonicalizing coordinate order."""
    if actual.batch_size != expected.batch_size:
        return False, f"batch_size mismatch: {actual.batch_size} vs {expected.batch_size}"
    if list(actual.spatial_shape) != list(expected.spatial_shape):
        return False, (
            f"spatial_shape mismatch: {actual.spatial_shape} vs {expected.spatial_shape}"
        )
    actual_idx, actual_feat = _sort_by_coords(actual.indices, actual.features, actual.spatial_shape)
    expected_idx, expected_feat = _sort_by_coords(expected.indices, expected.features, expected.spatial_shape)
    if not torch.equal(actual_idx, expected_idx):
        return False, "active coordinates differ"
    if not torch.allclose(actual_feat, expected_feat, atol=atol, rtol=rtol):
        return False, "active features differ"
    return True, "ok"


def _sort_by_coords(indices, features, spatial_shape):
    if indices.numel() == 0:
        return indices, features
    d, h, w = [int(v) for v in spatial_shape]
    keys = (
        ((indices[:, 0].to(torch.int64) * d + indices[:, 1].to(torch.int64)) * h
         + indices[:, 2].to(torch.int64)) * w + indices[:, 3].to(torch.int64)
    )
    order = torch.argsort(keys)
    return indices[order], features[order]


def reference_subm_conv3d(
    x: GTSparseSparseConvTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> GTSparseSparseConvTensor:
    """Reference SubMConv3d: dense conv3d then gather at active positions.

    Output has same active set and feature order as input.
    Index grid and worklist are shared with input.
    """
    dense = x.dense()
    out_dense = F.conv3d(
        dense, weight, bias, stride=1,
        padding=_normalize_3tuple(padding),
        dilation=_normalize_3tuple(dilation),
        groups=groups,
    )
    # Gather at input positions — output row i = conv result at input voxel i
    out_features = gather_dense_features_at_coords(out_dense, x.indices)
    out = x.replace_feature(out_features)
    return out


def reference_sparse_conv3d(
    x: GTSparseSparseConvTensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
    indice_key=None,
) -> GTSparseSparseConvTensor:
    """Reference SparseConv3d with geometry-defined active outputs.

    Output is scan-ordered (nonzero produces coords in raster order).
    """
    stride = _normalize_3tuple(stride)
    padding = _normalize_3tuple(padding)
    dilation = _normalize_3tuple(dilation)

    dense = x.dense()
    out_dense = F.conv3d(dense, weight, bias, stride=stride, padding=padding,
                         dilation=dilation, groups=groups)
    out_spatial = list(out_dense.shape[2:])

    input_mask = x.active_mask().to(dtype=dense.dtype).unsqueeze(1)
    out_mask = F.max_pool3d(
        input_mask, kernel_size=weight.shape[2:],
        stride=stride, padding=padding, dilation=dilation,
    ).squeeze(1) > 0

    out_indices = out_mask.nonzero(as_tuple=False).int()
    out_features = gather_dense_features_at_coords(out_dense, out_indices)

    out = GTSparseSparseConvTensor(
        out_features, out_indices, out_spatial, x.batch_size,
    )
    out.indice_dict = dict(x.indice_dict)
    if indice_key is not None:
        out.indice_dict[indice_key] = IndiceData(
            in_indices=x.indices, out_indices=out_indices,
            in_spatial_shape=x.spatial_shape, out_spatial_shape=out_spatial,
            kernel_size=tuple(int(v) for v in weight.shape[2:]),
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
    return out
