"""Additional sparse modules for spconv API compatibility."""

import torch.nn as nn
from .sparse_tensor import GTSparseSparseConvTensor


class GTSparseSparseSequential(nn.Sequential):
    """Drop-in replacement for spconv.SparseSequential.

    Passes GTSparseSparseConvTensor through child modules. For non-sparse
    modules (BatchNorm1d, ReLU, etc.), operates on .features directly.
    """

    def forward(self, x):
        for module in self:
            if isinstance(x, GTSparseSparseConvTensor):
                # Sparse conv modules accept and return GTSparseSparseConvTensor
                if hasattr(module, 'forward') and _is_sparse_module(module):
                    x = module(x)
                else:
                    # Non-sparse module (BN, ReLU, etc.) — apply to features
                    x = x.replace_feature(module(x.features))
            else:
                x = module(x)
        return x


def _is_sparse_module(module):
    """Check if a module expects GTSparseSparseConvTensor input."""
    from .conv_modules import (
        GTSparseSubMConv3d, GTSparseSparseConv3d, GTSparseSparseInverseConv3d
    )
    return isinstance(module, (
        GTSparseSubMConv3d, GTSparseSparseConv3d, GTSparseSparseInverseConv3d,
        GTSparseSparseSequential,
    ))
