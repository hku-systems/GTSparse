from __future__ import annotations

from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

from .metadata import GeometricTemplateMetadata, ensure_metadata


GeometricTemplateSparseTensor = GTSparseSparseConvTensor


def tensor_metadata(tensor: GTSparseSparseConvTensor) -> GeometricTemplateMetadata:
    return ensure_metadata(getattr(tensor, "metadata", None))
