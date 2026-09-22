import numpy as np
import torch
import cumm
import spconv.pytorch as spconv
import torchsparse
import torchsparse.nn as spnn
import MinkowskiEngine as ME

from gtsparse.sparse3d.geometric_template import GeometricTemplateSubMConv3d
from gtsparse.sparse3d.reference_ops import reference_subm_conv3d
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torchsparse.backends.allow_tf32 = False

torch.manual_seed(0)
grid = torch.cartesian_prod(
    torch.arange(1), torch.arange(1, 7), torch.arange(1, 7), torch.arange(1, 7)
).to(torch.int32)
coords = grid[torch.randperm(grid.size(0))[:96]].contiguous().cuda()
channels = 64
features = torch.randn(96, channels, device="cuda")


def canonical(feats, c):
    key = c.detach().cpu().numpy()
    order = np.lexsort((key[:, 3], key[:, 2], key[:, 1], key[:, 0]))
    return feats[torch.as_tensor(order, device=feats.device)]


def check_engine(name, out_feats, out_coords, expected_feats, atol, rtol):
    actual = canonical(out_feats, out_coords)
    target = canonical(expected_feats, coords)
    torch.testing.assert_close(actual, target, atol=atol, rtol=rtol)
    error = (actual - target).abs().max().item()
    print(f"  {name}: max_error={error:.6g} (atol={atol}, rtol={rtol})")


for dtype, atol, rtol in (
    (torch.float32, 1e-3, 1e-3),
    (torch.float16, 2e-2, 2e-2),
):
    print(dtype)
    feats = features.to(dtype)
    base_weight = torch.randn(27, channels, channels, device="cuda").to(dtype)

    gt = GeometricTemplateSubMConv3d(channels, channels, 3, padding=1).cuda().to(dtype)
    with torch.no_grad():
        gt.weight.copy_(base_weight)
    gt_sparse = GTSparseSparseConvTensor(feats, coords, (8, 8, 8), 1)
    with torch.no_grad():
        gt_out = gt(gt_sparse)
        weight_dense = base_weight.view(3, 3, 3, channels, channels).permute(4, 3, 0, 1, 2).contiguous()
        expected = reference_subm_conv3d(gt_sparse, weight_dense, None, padding=1)
    check_engine("gtsparse vs dense", gt_out.features, coords, expected.features, atol, rtol)

    sp = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False).cuda().to(dtype)
    with torch.no_grad():
        sp.weight.copy_(base_weight.view(3, 3, 3, channels, channels).permute(4, 0, 1, 2, 3))
    sp_in = spconv.SparseConvTensor(feats, coords, [8, 8, 8], 1)
    with torch.no_grad():
        sp_out = sp(sp_in)
    sp_atol = 1.0 if dtype == torch.float16 else atol
    check_engine("spconv vs dense", sp_out.features, sp_out.indices, expected.features, sp_atol, rtol)

    ts = spnn.Conv3d(channels, channels, kernel_size=3, stride=1, bias=False).cuda().to(dtype)
    with torch.no_grad():
        ts.kernel.copy_(base_weight.view(3, 3, 3, channels, channels).permute(2, 1, 0, 3, 4).reshape(27, channels, channels))
    ts_in = torchsparse.SparseTensor(feats, coords, spatial_range=(1, 8, 8, 8))
    with torch.no_grad():
        ts_out = ts(ts_in)
    check_engine("torchsparse vs dense", ts_out.F, ts_out.C, expected.features, atol, rtol)

    if dtype == torch.float32:
        me = ME.MinkowskiConvolution(channels, channels, kernel_size=3, stride=1, bias=False, dimension=3).cuda()
        with torch.no_grad():
            me.kernel.copy_(base_weight.view(3, 3, 3, channels, channels).permute(2, 1, 0, 3, 4).reshape(27, channels, channels))
        me_in = ME.SparseTensor(feats, coordinates=coords)
        with torch.no_grad():
            me_out = me(me_in)
        check_engine("minkowski vs dense", me_out.F, me_out.C, expected.features, atol, rtol)

ME.clear_global_coordinate_manager()
print("all cross-engine checks passed")
