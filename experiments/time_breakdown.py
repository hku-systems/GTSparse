import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from experiments.workloads import WORKLOADS, build_workload, move_batch, sparse_forward
import gtsparse.sparse3d.geometric_template.ops as gt_ops
import gtsparse.sparse3d.geometric_template.kernel3 as kernel3
import gtsparse.sparse3d.geometric_template.kernel8 as kernel8
import gtsparse.sparse3d.geometric_template.kernel9 as kernel9


builder_events = []
kernel_events = []
original_build_subm = gt_ops.build_subm_runtime_from_coords
original_build_full = gt_ops.build_full_runtime_from_coords
original_build_reverse = gt_ops.build_reverse_runtime_from_full_runtime
original_conv = gt_ops._conv
original_kernel3_build = kernel3.GeometricTemplateKernel3Conv3d.build_runtime
original_kernel3_conv = kernel3.kernel3_conv
original_kernel9_build = kernel9.GeometricTemplateKernel9Conv3d.build_runtime
original_kernel9_conv = kernel9.kernel9_conv
original_kernel8_build = kernel8.GeometricTemplateKernel8Conv3d.build_runtime
original_kernel8_reverse_build = kernel8.Kernel8ReverseSpec.build
original_kernel8_conv = kernel8._conv


def _timed_call(events, fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    events.append((start, end))
    return result


def timed_build_subm(*args, **kwargs):
    return _timed_call(builder_events, original_build_subm, *args, **kwargs)


def timed_build_full(*args, **kwargs):
    return _timed_call(builder_events, original_build_full, *args, **kwargs)


def timed_build_reverse(*args, **kwargs):
    return _timed_call(builder_events, original_build_reverse, *args, **kwargs)


def timed_conv(*args, **kwargs):
    return _timed_call(kernel_events, original_conv, *args, **kwargs)


def timed_kernel3_build(self, *args, **kwargs):
    return _timed_call(builder_events, original_kernel3_build, self, *args, **kwargs)


def timed_kernel9_build(self, *args, **kwargs):
    return _timed_call(builder_events, original_kernel9_build, self, *args, **kwargs)


def timed_kernel8_build(self, *args, **kwargs):
    return _timed_call(builder_events, original_kernel8_build, self, *args, **kwargs)


def timed_kernel8_reverse_build(self, *args, **kwargs):
    return _timed_call(builder_events, original_kernel8_reverse_build, self, *args, **kwargs)


def timed_kernel3_conv(*args, **kwargs):
    return _timed_call(kernel_events, original_kernel3_conv, *args, **kwargs)


def timed_kernel9_conv(*args, **kwargs):
    return _timed_call(kernel_events, original_kernel9_conv, *args, **kwargs)


def timed_kernel8_conv(*args, **kwargs):
    return _timed_call(kernel_events, original_kernel8_conv, *args, **kwargs)


def enable_native_timing() -> None:
    gt_ops.build_subm_runtime_from_coords = timed_build_subm
    gt_ops.build_full_runtime_from_coords = timed_build_full
    gt_ops.build_reverse_runtime_from_full_runtime = timed_build_reverse
    gt_ops._conv = timed_conv
    kernel3.GeometricTemplateKernel3Conv3d.build_runtime = timed_kernel3_build
    kernel3.kernel3_conv = timed_kernel3_conv
    kernel9.GeometricTemplateKernel9Conv3d.build_runtime = timed_kernel9_build
    kernel9.kernel9_conv = timed_kernel9_conv
    kernel8.GeometricTemplateKernel8Conv3d.build_runtime = timed_kernel8_build
    kernel8.Kernel8ReverseSpec.build = timed_kernel8_reverse_build
    kernel8._conv = timed_kernel8_conv


def elapsed_sum(events) -> float:
    return sum(float(start.elapsed_time(end)) for start, end in events)


def measure_native(model, batch, workload: str, device: str):
    builder_events.clear()
    kernel_events.clear()
    sparse_forward(model, batch, workload)
    torch.cuda.synchronize(device)
    builder = elapsed_sum(builder_events)
    kernel = elapsed_sum(kernel_events)
    return builder, kernel, builder + kernel


def measure_cache_difference(model, batch, workload: str, device: str):
    cold_start = torch.cuda.Event(enable_timing=True)
    cold_end = torch.cuda.Event(enable_timing=True)
    warm_start = torch.cuda.Event(enable_timing=True)
    warm_end = torch.cuda.Event(enable_timing=True)
    cold_start.record()
    sparse_forward(model, batch, workload)
    cold_end.record()
    warm_start.record()
    sparse_forward(model, batch, workload)
    warm_end.record()
    torch.cuda.synchronize(device)
    total = float(cold_start.elapsed_time(cold_end))
    kernel = float(warm_start.elapsed_time(warm_end))
    return total - kernel, kernel, total


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--backend", choices=("gtsparse", "spconv", "torchsparse", "minkowski"), required=True)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = "fp32" if args.backend == "minkowski" else args.dtype
    model, loader, runtime_dtype = build_workload(
        args.workload,
        args.backend,
        dtype,
        args.frames,
        args.device,
        random_sample=True,
    )
    if args.backend == "gtsparse":
        enable_native_timing()

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.warmup:
                break
            batch = move_batch(batch, args.device, runtime_dtype)
            sparse_forward(model, batch, args.workload)
            sparse_forward(model, batch, args.workload)
    torch.cuda.synchronize(args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    method = "native_events" if args.backend == "gtsparse" else "cold_minus_warm"
    with args.out.open("w", encoding="utf-8") as output:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"breakdown/{args.workload}/{args.backend}", dynamic_ncols=True):
                batch = move_batch(batch, args.device, runtime_dtype)
                if args.backend == "gtsparse":
                    builder_ms, kernel_ms, total_ms = measure_native(model, batch, args.workload, args.device)
                else:
                    builder_ms, kernel_ms, total_ms = measure_cache_difference(model, batch, args.workload, args.device)
                json.dump(
                    {
                        "backend": args.backend,
                        "builder_ms": builder_ms,
                        "dtype": dtype,
                        "frame_ids": list(batch.frame_ids),
                        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
                        "kernel_ms": kernel_ms,
                        "method": method,
                        "total_ms": total_ms,
                        "workload": args.workload,
                    },
                    output,
                    sort_keys=True,
                )
                output.write("\n")
                output.flush()


if __name__ == "__main__":
    main()
