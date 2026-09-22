#!/usr/bin/env python3

import argparse
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.workloads import build_workload, move_batch
from gtsparse import _C


TARGETS = {
    "second_kitti_sweeps1": (
        "sparse_backbone.conv_out.conv",
    ),
    "voxelnext_nuscenes_sweeps1": (
        "sparse_backbone.bev_tail.conv_out",
        "sparse_backbone.bev_tail.shared_conv",
    ),
    "voxelnext_nuscenes_sweeps10": (
        "sparse_backbone.bev_tail.conv_out",
        "sparse_backbone.bev_tail.shared_conv",
    ),
    "minkunet_semantickitti_sweeps1": (
        "sparse_backbone.down1.down",
        "sparse_backbone.down2.down",
        "sparse_backbone.down3.down",
        "sparse_backbone.down4.down",
        "sparse_backbone.up4.up",
        "sparse_backbone.up3.up",
        "sparse_backbone.up2.up",
        "sparse_backbone.up1.up",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=tuple(TARGETS), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def normalize_tuple(value, dimensions):
    if isinstance(value, int):
        return (value,) * dimensions
    return tuple(int(item) for item in value)


def sparse_features(tensor):
    if hasattr(tensor, "feats"):
        return tensor.feats
    return tensor.features


def sparse_coords(tensor):
    if hasattr(tensor, "coords"):
        return tensor.coords
    return tensor.indices


def mask_summary(active):
    volume = int(active.size(1))
    bits = 1 << torch.arange(volume, device=active.device, dtype=torch.int64)
    masks = (active.to(torch.int64) * bits).sum(dim=1)
    unique, counts = torch.unique(masks, return_counts=True)
    pairs = sorted(
        zip(unique.cpu().tolist(), counts.cpu().tolist()),
        key=lambda item: (-item[1], item[0]),
    )
    total = max(1, int(active.size(0)))
    return {
        "active_width_mean": float(active.sum().item()) / total,
        "mask_histogram": {str(mask): int(count) for mask, count in pairs},
    }


def kernel3_mask_summary(runtime):
    counts = runtime.template_counts.cpu().tolist()
    masks = (1, 2, 4, 3, 5, 6, 7)
    histogram = sorted(
        ((mask, int(count)) for mask, count in zip(masks, counts) if count),
        key=lambda item: (-item[1], item[0]),
    )
    total = max(1, sum(counts))
    return {
        "active_width_mean": sum(mask.bit_count() * count for mask, count in histogram) / total,
        "mask_histogram": {str(mask): count for mask, count in histogram},
    }


def kernel3_latency(module, input_tensor, runtime, repeats):
    fn = (
        _C.gtsparse_kernel3_fp16_forward
        if sparse_features(input_tensor).dtype == torch.float16
        else _C.gtsparse_kernel3_fp32_forward
    )
    events = []
    for _ in range(int(repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(
            sparse_features(input_tensor),
            module._runtime_weight(),
            runtime.out_rows,
            runtime.input_rows_w1,
            runtime.input_rows_w2,
            runtime.input_rows_w3,
            runtime.template_ids,
            runtime.input_row_offsets,
            runtime.out_coords.size(0),
        )
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(sparse_features(input_tensor).device)
    return statistics.median(start.elapsed_time(end) for start, end in events)


def kernel9_active_map(module, input_tensor, output_tensor):
    input_coords = sparse_coords(input_tensor).to(torch.int64)
    output_coords = sparse_coords(output_tensor).to(torch.int64)
    spatial = torch.tensor(tuple(input_tensor.spatial_shape), device=input_coords.device)
    input_keys = (
        ((input_coords[:, 0] * spatial[0] + input_coords[:, 1]) * spatial[1] + input_coords[:, 2])
        * spatial[2]
        + input_coords[:, 3]
    )
    sorted_keys, _ = torch.sort(input_keys)
    offsets = torch.cartesian_prod(
        torch.arange(3, device=input_coords.device),
        torch.arange(3, device=input_coords.device),
    )
    offsets = torch.cat((offsets, torch.zeros((9, 1), device=input_coords.device)), dim=1)
    query = (
        output_coords[:, None, 1:] * torch.tensor(module.stride, device=input_coords.device)
        - torch.tensor(module.padding, device=input_coords.device)
        + offsets[None] * torch.tensor(module.dilation, device=input_coords.device)
    )
    valid = ((query >= 0) & (query < spatial)).all(dim=2)
    batches = output_coords[:, None, 0].expand(-1, 9)
    query_keys = ((batches * spatial[0] + query[:, :, 0]) * spatial[1] + query[:, :, 1]) * spatial[2] + query[:, :, 2]
    positions = torch.searchsorted(sorted_keys, query_keys.flatten())
    safe = positions.clamp_max(sorted_keys.numel() - 1)
    found = positions.lt(sorted_keys.numel()) & sorted_keys[safe].eq(query_keys.flatten())
    return (valid.flatten() & found).view(output_coords.size(0), 9)


def kernel9_latency(module, input_tensor, runtime, repeats):
    fn = (
        _C.gtsparse_kernel9_fp16_forward
        if sparse_features(input_tensor).dtype == torch.float16
        else _C.gtsparse_kernel9_fp32_forward
    )
    events = []
    for _ in range(int(repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(
            sparse_features(input_tensor),
            module._runtime_weight(),
            runtime.out_rows,
            runtime.input_rows_w1,
            runtime.input_rows_w4,
            runtime.input_rows_w7,
            runtime.input_rows_w9,
            runtime.template_ids,
            runtime.input_row_offsets,
            runtime.out_coords.size(0),
        )
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(sparse_features(input_tensor).device)
    return statistics.median(start.elapsed_time(end) for start, end in events)


def kernel8_forward_active_map(module, input_tensor, output_tensor):
    input_coords = sparse_coords(input_tensor).to(torch.int64)
    output_coords = sparse_coords(output_tensor).to(torch.int64)
    spatial = torch.tensor(tuple(input_tensor.spatial_shape), device=input_coords.device)
    input_keys = ((input_coords[:, 0] * spatial[0] + input_coords[:, 1]) * spatial[1] + input_coords[:, 2]) * spatial[2] + input_coords[:, 3]
    sorted_keys, _ = torch.sort(input_keys)
    offsets = torch.cartesian_prod(
        torch.arange(2, device=input_coords.device),
        torch.arange(2, device=input_coords.device),
        torch.arange(2, device=input_coords.device),
    )
    query = output_coords[:, None, 1:] * 2 + offsets[None]
    batches = output_coords[:, None, 0].expand(-1, 8)
    query_keys = ((batches * spatial[0] + query[:, :, 0]) * spatial[1] + query[:, :, 1]) * spatial[2] + query[:, :, 2]
    positions = torch.searchsorted(sorted_keys, query_keys.flatten())
    safe = positions.clamp_max(sorted_keys.numel() - 1)
    found = positions.lt(sorted_keys.numel()) & sorted_keys[safe].eq(query_keys.flatten())
    return found.view(output_coords.size(0), 8)


def kernel8_reverse_summary(runtime):
    counts = runtime.template_counts.cpu().tolist()
    histogram = [(0, counts[0])]
    histogram.extend((1 << phase, counts[phase + 1]) for phase in range(8))
    histogram = sorted(((mask, count) for mask, count in histogram if count), key=lambda item: (-item[1], item[0]))
    total = sum(count for _, count in histogram)
    return {
        "active_width_mean": sum(mask.bit_count() * count for mask, count in histogram) / total,
        "mask_histogram": {str(mask): count for mask, count in histogram},
    }


def kernel8_latency(module, input_tensor, runtime, repeats):
    fn = _C.gtsparse_kernel8_fp16_forward if sparse_features(input_tensor).dtype == torch.float16 else _C.gtsparse_kernel8_fp32_forward
    events = []
    for _ in range(int(repeats)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(
            sparse_features(input_tensor), module._runtime_weight(), runtime.out_rows,
            runtime.input_rows_w1, runtime.input_rows_w2, runtime.input_rows_w4,
            runtime.input_rows_w8, runtime.template_ids, runtime.input_row_offsets,
            runtime.out_coords.size(0),
        )
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(sparse_features(input_tensor).device)
    return statistics.median(start.elapsed_time(end) for start, end in events)


def torchsparse_active_map(module, tensor):
    kernel = normalize_tuple(module.kernel_size, 3)
    stride = normalize_tuple(module.stride, 3)
    dilation = normalize_tuple(module.dilation, 3)
    tensor_stride = tuple(int(value) for value in tensor.stride)
    if module.transposed:
        tensor_stride = tuple(tensor_stride[dim] // stride[dim] for dim in range(3))
    key = (tensor_stride, kernel, stride, dilation)
    kmap = tensor._caches.kmaps[key]
    n_in, n_out = (int(value) for value in kmap["sizes"])
    forward_rows = kmap["out_in_map"][:n_out, : math.prod(kernel)]
    if not module.transposed:
        return forward_rows.ge(0)
    reverse = torch.zeros((n_in, forward_rows.size(1)), device=forward_rows.device, dtype=torch.bool)
    slots = torch.arange(forward_rows.size(1), device=forward_rows.device)
    slots = slots.view(1, -1).expand_as(forward_rows)
    valid = forward_rows.ge(0)
    reverse[forward_rows[valid].long(), slots[valid]] = True
    return reverse


def spconv_active_map(module, input_tensor, output_tensor):
    kernel = normalize_tuple(module.kernel_size, 2)
    stride = normalize_tuple(module.stride, 2)
    padding = normalize_tuple(module.padding, 2)
    dilation = normalize_tuple(module.dilation, 2)
    input_coords = input_tensor.indices.to(torch.int64)
    output_coords = output_tensor.indices.to(torch.int64)
    height, width = (int(value) for value in input_tensor.spatial_shape)
    input_keys = (input_coords[:, 0] * height + input_coords[:, 1]) * width + input_coords[:, 2]
    sorted_keys, _ = torch.sort(input_keys)
    offsets = torch.cartesian_prod(
        torch.arange(kernel[0], device=input_coords.device),
        torch.arange(kernel[1], device=input_coords.device),
    )
    query_xy = (
        output_coords[:, None, 1:] * torch.tensor(stride, device=input_coords.device)
        - torch.tensor(padding, device=input_coords.device)
        + offsets[None] * torch.tensor(dilation, device=input_coords.device)
    )
    valid = ((query_xy >= 0) & (query_xy < torch.tensor((height, width), device=input_coords.device))).all(dim=2)
    batches = output_coords[:, None, 0].expand(-1, offsets.size(0))
    query_keys = (batches * height + query_xy[:, :, 0]) * width + query_xy[:, :, 1]
    positions = torch.searchsorted(sorted_keys, query_keys.flatten())
    safe = positions.clamp_max(max(0, int(sorted_keys.numel()) - 1))
    found = positions.lt(sorted_keys.numel())
    found &= sorted_keys.index_select(0, safe) == query_keys.flatten()
    return (valid.flatten() & found).view(output_coords.size(0), offsets.size(0))


def module_description(module):
    kernel = tuple(int(value) for value in module.kernel_size)
    dimensions = len(kernel)
    return {
        "backend": module.__module__.split(".", 1)[0],
        "kernel_size": kernel,
        "stride": normalize_tuple(module.stride, dimensions),
        "transposed": bool(getattr(module, "transposed", False)),
        "in_channels": int(module.in_channels),
        "out_channels": int(module.out_channels),
    }


def profile_forward(model, workload, voxel_features, voxel_coords, batch_size):
    if workload.startswith("voxelnext_"):
        return model.sparse_backbone(voxel_features, voxel_coords, batch_size)
    return model.forward_sparse_backbone(voxel_features, voxel_coords, batch_size)


def print_summary(result):
    print(f"GPU: {result['gpu']}")
    print(f"Workload: {result['workload']}")
    print(f"Sample index: {result['sample_index']}")
    for layer in result["layers"]:
        kernel = "x".join(str(value) for value in layer["kernel_size"])
        stride = "x".join(str(value) for value in layer["stride"])
        print(
            f"\n{layer['name']} ({layer['backend']}, kernel={kernel}, stride={stride}, "
            f"transposed={layer['transposed']})"
        )
        print(
            f"  channels: {layer['in_channels']} -> {layer['out_channels']}; "
            f"rows: {layer['input_rows']} -> {layer['output_rows']}"
        )
        print(
            f"  latency: median={layer['latency_median_ms']:.5f} ms, "
            f"mean={layer['latency_mean_ms']:.5f} ms"
        )
        if "kernel_latency_median_ms" in layer:
            print(
                f"  native split: builder={layer['builder_latency_median_ms']:.5f} ms, "
                f"kernel={layer['kernel_latency_median_ms']:.5f} ms"
            )
        if "template_width_mean" in layer:
            print(
                f"  template width: {layer['template_width_mean']:.5f}; "
                f"counts={layer['template_counts']}"
            )
        print(
            f"  active width: {layer['active_width_mean']:.5f} / "
            f"{math.prod(layer['kernel_size'])}; masks={len(layer['mask_histogram'])}"
        )
        top = list(layer["mask_histogram"].items())[:16]
        print("  top masks: " + ", ".join(f"{mask}:{count}" for mask, count in top))


def main():
    args = parse_args()
    model, loader, dtype = build_workload(
        args.workload, "gtsparse", "fp16", int(args.sample_index) + 1, args.device
    )
    batch = move_batch(next(itertools.islice(loader, int(args.sample_index), None)), args.device, dtype)
    modules = dict(model.named_modules())
    captures = {
        name: {"module": modules[name], "events": []}
        for name in TARGETS[args.workload]
    }
    handles = []

    for name, capture in captures.items():
        def pre_hook(module, inputs, layer=name):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            captures[layer]["input"] = inputs[0]
            captures[layer]["start"] = start

        def post_hook(module, inputs, output, layer=name):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            captures[layer]["output"] = output
            captures[layer]["events"].append((captures[layer]["start"], end))

        handles.append(capture["module"].register_forward_pre_hook(pre_hook))
        handles.append(capture["module"].register_forward_hook(post_hook))

    voxel_features, voxel_coords, batch_size = model.encode_batch(batch)
    with torch.inference_mode():
        for _ in range(int(args.warmup) + int(args.repeats)):
            profile_forward(model, args.workload, voxel_features, voxel_coords, batch_size)
    torch.cuda.synchronize(args.device)

    layers = []
    for name, capture in captures.items():
        module = capture["module"]
        input_tensor = capture["input"]
        output_tensor = capture["output"]
        elapsed = [start.elapsed_time(end) for start, end in capture["events"][-int(args.repeats):]]
        if module.__module__.endswith(".kernel3"):
            runtime, _ = module.build_runtime(input_tensor)
            summary = kernel3_mask_summary(runtime)
        elif module.__module__.endswith(".kernel9"):
            runtime, _ = module.build_runtime(input_tensor)
            active = kernel9_active_map(module, input_tensor, output_tensor)
            summary = mask_summary(active)
        elif module.__module__.endswith(".kernel8"):
            if module.transposed:
                runtime = input_tensor.metadata.reverse_chain[0].runtime.build()
                summary = kernel8_reverse_summary(runtime)
            else:
                runtime, _ = module.build_runtime(input_tensor)
                active = kernel8_forward_active_map(module, input_tensor, output_tensor)
                summary = mask_summary(active)
        elif module.__module__.startswith("torchsparse"):
            active = torchsparse_active_map(module, input_tensor)
            summary = mask_summary(active)
        else:
            active = spconv_active_map(module, input_tensor, output_tensor)
            summary = mask_summary(active)
        layer = {
            "name": name,
            **module_description(module),
            "input_rows": int(sparse_features(input_tensor).size(0)),
            "output_rows": int(sparse_features(output_tensor).size(0)),
            "latency_median_ms": float(statistics.median(elapsed)),
            "latency_mean_ms": float(statistics.mean(elapsed)),
            **summary,
        }
        if module.__module__.endswith(".kernel3"):
            kernel_ms = kernel3_latency(module, input_tensor, runtime, args.repeats)
            layer["kernel_latency_median_ms"] = kernel_ms
            layer["builder_latency_median_ms"] = layer["latency_median_ms"] - kernel_ms
        elif module.__module__.endswith(".kernel9"):
            kernel_ms = kernel9_latency(module, input_tensor, runtime, args.repeats)
            counts = runtime.template_counts.cpu().tolist()
            widths = (1, 4, 3, 4, 6, 7, 6, 9)
            layer["kernel_latency_median_ms"] = kernel_ms
            layer["builder_latency_median_ms"] = layer["latency_median_ms"] - kernel_ms
            layer["template_counts"] = counts
            layer["template_width_mean"] = sum(
                count * width for count, width in zip(counts, widths)
            ) / sum(counts)
        elif module.__module__.endswith(".kernel8"):
            kernel_ms = kernel8_latency(module, input_tensor, runtime, args.repeats)
            counts = runtime.template_counts.cpu().tolist()
            widths = (0, *([1] * 8), *([4] * 6), 8)
            layer["kernel_latency_median_ms"] = kernel_ms
            layer["builder_latency_median_ms"] = layer["latency_median_ms"] - kernel_ms
            layer["template_counts"] = counts
            layer["template_width_mean"] = sum(
                count * width for count, width in zip(counts, widths)
            ) / sum(counts)
        layers.append(layer)

    for handle in handles:
        handle.remove()
    result = {
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "workload": args.workload,
        "sample_index": int(args.sample_index),
        "layers": layers,
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print_summary(result)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
