import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from experiments.workloads import WORKLOADS, build_workload, move_batch, sparse_forward


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--backend", choices=("gtsparse", "spconv", "torchsparse", "minkowski"), required=True)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = "fp32" if args.backend == "minkowski" else args.dtype
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    model, loader, runtime_dtype = build_workload(
        args.workload,
        args.backend,
        dtype,
        args.frames,
        args.device,
        random_sample=True,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.warmup:
                break
            sparse_forward(model, move_batch(batch, args.device, runtime_dtype), args.workload)
        for batch in tqdm(loader, desc=f"peak-memory/{args.workload}/{args.backend}", dynamic_ncols=True):
            sparse_forward(model, move_batch(batch, args.device, runtime_dtype), args.workload)
    torch.cuda.synchronize(args.device)

    result = {
        "backend": args.backend,
        "dtype": dtype,
        "frames": len(loader),
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "workload": args.workload,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
