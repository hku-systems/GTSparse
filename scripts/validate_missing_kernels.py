#!/usr/bin/env python3

import argparse

import torch

from gtsparse.sparse3d.geometric_template import (
    GeometricTemplateKernel3Conv3d,
    GeometricTemplateKernel8Conv3d,
    GeometricTemplateKernel8InverseConv3d,
    GeometricTemplateKernel9Conv3d,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def linear_keys(coords, spatial):
    depth, height, width = (int(value) for value in spatial)
    coords = coords.to(torch.int64)
    return ((coords[:, 0] * depth + coords[:, 1]) * height + coords[:, 2]) * width + coords[:, 3]


def reference_kernel3(features, weight, input_coords, output_coords, input_spatial):
    sorted_keys, order = torch.sort(linear_keys(input_coords, input_spatial))
    output = torch.zeros(
        (output_coords.size(0), weight.size(2)),
        device=features.device,
        dtype=features.dtype,
    )
    for offset in range(3):
        query_coords = output_coords.clone()
        query_coords[:, 1] = query_coords[:, 1] * 2 + offset
        query_keys = linear_keys(query_coords, input_spatial)
        positions = torch.searchsorted(sorted_keys, query_keys)
        safe = positions.clamp_max(sorted_keys.numel() - 1)
        found = positions.lt(sorted_keys.numel()) & sorted_keys[safe].eq(query_keys)
        input_rows = order[safe[found]]
        output[found] += features[input_rows] @ weight[offset]
    return output


def reference_kernel9(features, weight, input_coords, output_coords, input_spatial, bias):
    sorted_keys, order = torch.sort(linear_keys(input_coords, input_spatial))
    output = torch.zeros(
        (output_coords.size(0), weight.size(2)),
        device=features.device,
        dtype=features.dtype,
    )
    for offset in range(9):
        rd, rh = divmod(offset, 3)
        query_coords = output_coords.clone()
        query_coords[:, 1] += rd - 1
        query_coords[:, 2] += rh - 1
        query_keys = linear_keys(query_coords, input_spatial)
        positions = torch.searchsorted(sorted_keys, query_keys)
        safe = positions.clamp_max(sorted_keys.numel() - 1)
        found = positions.lt(sorted_keys.numel()) & sorted_keys[safe].eq(query_keys)
        output[found] += features[order[safe[found]]] @ weight[offset]
    if bias is not None:
        output += bias
    return output


def reference_kernel8(features, weight, input_coords, output_coords, input_spatial, inverse=False):
    sorted_keys, order = torch.sort(linear_keys(input_coords, input_spatial))
    output = torch.zeros((output_coords.size(0), weight.size(2)), device=features.device, dtype=features.dtype)
    spatial = torch.tensor(input_spatial, device=features.device)
    for offset in range(8):
        rd, remain = divmod(offset, 4)
        rh, rw = divmod(remain, 2)
        query = output_coords.clone()
        if inverse:
            query[:, 1] -= rd
            query[:, 2] -= rh
            query[:, 3] -= rw
            valid = (query[:, 1:].remainder(2) == 0).all(dim=1)
            query[:, 1:] = torch.div(query[:, 1:], 2, rounding_mode="floor")
            valid &= ((query[:, 1:] >= 0) & (query[:, 1:] < spatial)).all(dim=1)
        else:
            query[:, 1] = query[:, 1] * 2 + rd
            query[:, 2] = query[:, 2] * 2 + rh
            query[:, 3] = query[:, 3] * 2 + rw
            valid = ((query[:, 1:] >= 0) & (query[:, 1:] < spatial)).all(dim=1)
        query_keys = linear_keys(query, input_spatial)
        positions = torch.searchsorted(sorted_keys, query_keys)
        safe = positions.clamp_max(sorted_keys.numel() - 1)
        found = valid & positions.lt(sorted_keys.numel()) & sorted_keys[safe].eq(query_keys)
        output[found] += features[order[safe[found]]] @ weight[offset]
    return output


def validate_stream(runtime, output_rows):
    padded_rows = int(runtime.padded_counts.sum().item())
    live_out_rows = runtime.out_rows[:padded_rows]
    live_out_rows = live_out_rows[live_out_rows.ge(0)]
    assert live_out_rows.numel() == output_rows
    assert torch.unique(live_out_rows).numel() == output_rows
    assert runtime.out_rows[padded_rows:].eq(-1).all()
    assert runtime.template_ids[padded_rows:].eq(-1).all()


def validate_dtype(dtype, device):
    repeats = 300
    input_spatial = (6, repeats * 7 + 1, 1)
    coords = []
    expected_coords = set()
    for repeat in range(repeats):
        for mask in range(1, 8):
            height = repeat * 7 + mask
            for offset in range(3):
                if mask & (1 << offset):
                    coords.append((0, 2 + offset, height, 0))
            expected_coords.add((0, 1, height, 0))
            if mask & 1:
                expected_coords.add((0, 0, height, 0))

    coords = torch.tensor(sorted(coords), device=device, dtype=torch.int32)
    expected_coords = torch.tensor(sorted(expected_coords), device=device, dtype=torch.int32)
    features = (torch.randn(coords.size(0), 64, device=device) * 0.1).to(dtype)
    module = GeometricTemplateKernel3Conv3d(64, 128, stride=(2, 1, 1)).to(device=device, dtype=dtype).eval()
    sparse_input = GTSparseSparseConvTensor(features, coords, input_spatial, 1)
    with torch.inference_mode():
        output = module(sparse_input)

    runtime, _ = module.build_runtime(sparse_input)
    expected_counts = [repeats, repeats, repeats * 5, repeats, repeats, repeats, repeats]
    assert runtime.template_counts.cpu().tolist() == expected_counts
    assert torch.equal(output.indices, expected_coords)

    validate_stream(runtime, output.features.size(0))

    expected_features = reference_kernel3(
        features,
        module.weight.detach(),
        coords,
        output.indices,
        input_spatial,
    )
    error = (output.features - expected_features).abs().max().item()
    tolerance = 1e-5 if dtype == torch.float32 else 2e-3
    assert error <= tolerance, f"{dtype} max error {error} exceeds {tolerance}"
    average_width = sum(
        count * (1 if template < 3 else 2 if template < 6 else 3)
        for template, count in enumerate(expected_counts)
    ) / sum(expected_counts)
    print(f"{dtype} rows={output.features.size(0)} average_width={average_width:.5f} max_error={error:.9g}")


def validate_kernel9_dtype(dtype, device):
    patterns = (
        (4,),
        (0, 3, 4, 6),
        (1, 4, 7),
        (2, 4, 5, 8),
        (1, 2, 4, 5, 7, 8),
        (0, 2, 3, 4, 5, 6, 8),
        (0, 1, 3, 4, 6, 7),
        tuple(range(9)),
    )
    repeats = 140
    input_spatial = (7, repeats * len(patterns) * 6 + 5, 1)
    coords = set()
    for repeat in range(repeats):
        for pattern_index, pattern in enumerate(patterns):
            center_d = 3
            center_h = 3 + (repeat * len(patterns) + pattern_index) * 6
            for offset in pattern:
                rd, rh = divmod(offset, 3)
                coords.add((0, center_d + rd - 1, center_h + rh - 1, 0))
    coords = torch.tensor(sorted(coords), device=device, dtype=torch.int32)
    features = (torch.randn(coords.size(0), 128, device=device) * 0.1).to(dtype)
    sparse_input = GTSparseSparseConvTensor(features, coords, input_spatial, 1)
    errors = []

    for subm in (False, True):
        module = GeometricTemplateKernel9Conv3d(
            128,
            128,
            bias=subm,
            subm=subm,
        ).to(device=device, dtype=dtype).eval()
        with torch.inference_mode():
            output = module(sparse_input)
        runtime, _ = module.build_runtime(sparse_input)
        validate_stream(runtime, output.features.size(0))
        assert all(count > 0 for count in runtime.template_counts.cpu().tolist())
        if subm:
            assert torch.equal(output.indices, coords)
        expected_features = reference_kernel9(
            features,
            module.weight.detach(),
            coords,
            output.indices,
            input_spatial,
            module.bias,
        )
        error = (output.features - expected_features).abs().max().item()
        tolerance = 1e-5 if dtype == torch.float32 else 2e-3
        assert error <= tolerance, f"kernel9 {dtype} subm={subm} max error {error} exceeds {tolerance}"
        errors.append(error)
    print(
        f"kernel9 {dtype} input_rows={coords.size(0)} "
        f"regular_error={errors[0]:.9g} subm_error={errors[1]:.9g}"
    )


def validate_kernel8_dtype(dtype, channels, device):
    spatial = (25, 25, 25)
    coords = torch.tensor(
        [
            (0, d, h, w)
            for d in range(spatial[0])
            for h in range(spatial[1])
            for w in range(spatial[2])
            if (d * 3 + h * 5 + w * 7) % 17 < 3
        ],
        device=device,
        dtype=torch.int32,
    )
    features = (torch.randn(coords.size(0), channels, device=device) * 0.1).to(dtype)
    sparse_input = GTSparseSparseConvTensor(features, coords, spatial, 1)
    down = GeometricTemplateKernel8Conv3d(channels, channels).to(device=device, dtype=dtype).eval()
    with torch.inference_mode():
        down_output = down(sparse_input)
    runtime, _ = down.build_runtime(sparse_input)
    validate_stream(runtime, down_output.features.size(0))
    down_reference = reference_kernel8(
        features, down.weight.detach(), coords, down_output.indices, spatial
    )
    down_error = (down_output.features - down_reference).abs().max().item()

    inverse_features = (torch.randn_like(down_output.features.float()) * 0.1).to(dtype)
    inverse = GeometricTemplateKernel8InverseConv3d(channels, channels).to(device=device, dtype=dtype).eval()
    inverse_input = down_output.replace_feature(inverse_features)
    reverse_runtime = inverse_input.metadata.reverse_chain[0].runtime.build()
    with torch.inference_mode():
        inverse_output = inverse(inverse_input)
    validate_stream(reverse_runtime, inverse_output.features.size(0))
    inverse_reference = reference_kernel8(
        inverse_features,
        inverse.weight.detach(),
        down_output.indices,
        inverse_output.indices,
        down_output.spatial_shape,
        inverse=True,
    )
    inverse_error = (inverse_output.features - inverse_reference).abs().max().item()
    tolerance = 1e-5 if dtype == torch.float32 else 2e-3
    assert down_error <= tolerance
    assert inverse_error <= tolerance
    print(
        f"kernel8 {dtype} channels={channels} input_rows={coords.size(0)} "
        f"down_error={down_error:.9g} inverse_error={inverse_error:.9g}"
    )


def main():
    args = parse_args()
    torch.manual_seed(0)
    validate_dtype(torch.float32, args.device)
    validate_dtype(torch.float16, args.device)
    validate_kernel9_dtype(torch.float32, args.device)
    validate_kernel9_dtype(torch.float16, args.device)
    validate_kernel8_dtype(torch.float32, 32, args.device)
    validate_kernel8_dtype(torch.float32, 64, args.device)
    validate_kernel8_dtype(torch.float16, 32, args.device)
    validate_kernel8_dtype(torch.float16, 64, args.device)


if __name__ == "__main__":
    main()
