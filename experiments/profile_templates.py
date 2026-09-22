import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from experiments.workloads import WORKLOADS, build_workload, move_batch, sparse_forward
from gtsparse.sparse3d.geometric_template import (
    GeometricTemplateSparseConv3d,
    GeometricTemplateSparseInverseConv3d,
    GeometricTemplateSubMConv3d,
)
import gtsparse.sparse3d.geometric_template.ops as gt_ops


capture = False
current_kind = None
current_counts = []
original_conv = gt_ops._conv


def module_pre_hook(module, inputs):
    global current_kind
    current_kind = "kernel27"


def module_hook(module, inputs, output):
    global current_kind
    current_kind = None


def observed_conv(features, logical_weight, runtime):
    if capture and current_kind == "kernel27":
        current_counts.append(runtime.template_counts)
    return original_conv(features, logical_weight, runtime)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    global capture
    args = parse_args()
    gt_ops._conv = observed_conv
    model, loader, dtype = build_workload(
        args.workload,
        "gtsparse",
        "fp16",
        args.frames,
        args.device,
        random_sample=True,
    )
    handles = []
    for module in model.modules():
        if isinstance(module, (
            GeometricTemplateSubMConv3d,
            GeometricTemplateSparseConv3d,
            GeometricTemplateSparseInverseConv3d,
        )):
            handles.append(module.register_forward_pre_hook(module_pre_hook))
            handles.append(module.register_forward_hook(module_hook))

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
            for batch in tqdm(loader, desc=f"template-distribution/{args.workload}", dynamic_ncols=True):
                current_counts.clear()
                batch = move_batch(batch, args.device, dtype)
                sparse_forward(model, batch, args.workload)
                counts = torch.stack(current_counts).sum(dim=0).cpu().tolist()
                family_counts = (
                    counts[0],
                    sum(counts[1:4]),
                    sum(counts[4:7]),
                    counts[7],
                )
                record = {
                    "family_counts": family_counts,
                    "frame_ids": list(batch.frame_ids),
                    "gpu": torch.cuda.get_device_name(torch.device(args.device)),
                    "operator": "3x3x3",
                    "workload": args.workload,
                }
                json.dump(record, output, sort_keys=True)
                output.write("\n")
                output.flush()

    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
