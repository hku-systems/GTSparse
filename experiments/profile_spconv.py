import argparse
import json
from pathlib import Path

import torch
import spconv.pytorch.ops as spconv_ops
from tqdm.auto import tqdm

from experiments.workloads import WORKLOADS, build_workload, move_batch, sparse_forward


capture = False
current_layers = []
original_implicit_gemm = spconv_ops.implicit_gemm


def implicit_gemm(*args, **kwargs):
    result = original_implicit_gemm(*args, **kwargs)
    if capture:
        features = args[0]
        filters = args[1]
        current_layers.append(
            {
                "cin": int(features.size(1)),
                "cout": int(filters.size(0)),
                "n_out": int(args[5]),
                "tile_rows": int(result[2]),
            }
        )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first-frames", action="store_true", help="profile the first N frames instead of a random sample")
    return parser.parse_args()


def main() -> None:
    global capture
    args = parse_args()
    spconv_ops.implicit_gemm = implicit_gemm
    model, loader, dtype = build_workload(
        args.workload,
        "spconv",
        "fp16",
        args.frames,
        args.device,
        random_sample=not args.first_frames,
    )

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= args.warmup:
                break
            sparse_forward(model, move_batch(batch, args.device, dtype), args.workload)
    torch.cuda.synchronize(args.device)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    capture = True
    with args.out.open("w", encoding="utf-8") as output:
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"spconv-profile/{args.workload}", dynamic_ncols=True):
                current_layers.clear()
                batch = move_batch(batch, args.device, dtype)
                sparse_forward(model, batch, args.workload)
                torch.cuda.synchronize(args.device)
                json.dump(
                    {
                        "frame_ids": list(batch.frame_ids),
                        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
                        "layers": current_layers,
                        "workload": args.workload,
                    },
                    output,
                    sort_keys=True,
                )
                output.write("\n")
                output.flush()


if __name__ == "__main__":
    main()
