from __future__ import annotations

import statistics

import torch


def clear_sparse_metadata(backend: str) -> None:
    if backend == "torchsparse":
        from torchsparse.utils.tensor_cache import clear_global_tensor_cache

        clear_global_tensor_cache()
    elif backend == "minkowski":
        import MinkowskiEngine as ME

        ME.clear_global_coordinate_manager()


def measure_cuda_elapsed_ms(
    fn,
    *args,
    device: torch.device,
    repeats: int = 1,
    warmup_repeats: int = 0,
    clear_metadata=None,
    **kwargs,
):
    repeat_count = max(1, int(repeats))
    warmup_count = max(0, int(warmup_repeats))
    out = None
    for _ in range(warmup_count):
        out = fn(*args, **kwargs)
        torch.cuda.synchronize(device=device)
        if clear_metadata is not None:
            clear_metadata()
        out = None
    times = []
    for repeat_index in range(repeat_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize(device=device)
        times.append(float(start.elapsed_time(end)))
        if clear_metadata is not None:
            clear_metadata()
        if repeat_index + 1 < repeat_count:
            out = None
    return out, float(statistics.median(times))
