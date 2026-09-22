import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from experiments.workloads import WORKLOADS, build_workload, move_batch, sparse_forward
from gtsparse.sparse3d.geometric_template.runtime import PAYLOAD_LOGICAL_TO_ACTUAL, TEMPLATE_KEEP_SLOTS, TEMPLATE_SLOT_COUNTS
import gtsparse.sparse3d.geometric_template.ops as gt_ops
from gtsparse.sparse3d.geometric_template import (
    GeometricTemplateKernel3Conv3d,
    GeometricTemplateKernel8Conv3d,
    GeometricTemplateKernel8InverseConv3d,
    GeometricTemplateKernel9Conv3d,
    GeometricTemplateSparseConv3d,
    GeometricTemplateSparseInverseConv3d,
    GeometricTemplateSubMConv3d,
)


capture = False
current_layers = []
current_kernel27_kind = None
original_conv = gt_ops._conv
TORCHSPARSE_BM = 128

K3_OFFSETS = ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2))
K9_OFFSETS = (
    (4,), (0, 3, 4, 6), (1, 4, 7), (2, 4, 5, 8),
    (1, 2, 4, 5, 7, 8), (0, 2, 3, 4, 5, 6, 8),
    (0, 1, 3, 4, 6, 7), tuple(range(9)),
)
K8_OFFSETS = (
    (), *((offset,) for offset in range(8)),
    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 4, 5),
    (2, 3, 6, 7), (0, 2, 4, 6), (1, 3, 5, 7), tuple(range(8)),
)


def _template_buffer(runtime, template_id: int, count: int):
    if template_id == 0:
        return runtime.input_rows_w1[0, :count, :1]
    if template_id < 4:
        return runtime.input_rows_w9[template_id - 1, :count, : len(TEMPLATE_KEEP_SLOTS[template_id])]
    if template_id < 7:
        return runtime.input_rows_w18[template_id - 4, :count, : len(TEMPLATE_KEEP_SLOTS[template_id])]
    return runtime.input_rows_w27[0, :count, :27]


def _runtime_masks(runtime, counts: list[int]) -> np.ndarray:
    all_masks = []
    for template_id, count in enumerate(counts):
        if count == 0:
            continue
        buffer = _template_buffer(runtime, template_id, count)
        actual_offsets = [PAYLOAD_LOGICAL_TO_ACTUAL[slot] for slot in TEMPLATE_KEEP_SLOTS[template_id]]
        bits = torch.tensor([1 << offset for offset in actual_offsets], device=buffer.device, dtype=torch.int64)
        masks = ((buffer >= 0).to(torch.int64) * bits).sum(dim=1).cpu().numpy().astype(np.uint32)
        all_masks.append(masks)
    return np.concatenate(all_masks) if all_masks else np.empty(0, dtype=np.uint32)


def _spconv_offset_rows(masks: np.ndarray, tile_rows: int) -> int:
    masks = np.sort(masks)
    issued = 0
    for start in range(0, len(masks), tile_rows):
        tile = masks[start : start + tile_rows]
        issued += tile_rows * int(int(np.bitwise_or.reduce(tile)).bit_count())
    return issued


def _padded_rows(rows: int, tile_rows: int) -> int:
    return ((int(rows) + int(tile_rows) - 1) // int(tile_rows)) * int(tile_rows)


def observe_runtime(kind, features, logical_weight, runtime) -> None:
    counts = [int(value) for value in runtime.template_counts.tolist()]
    padded_counts = [int(value) for value in runtime.padded_counts.tolist()]
    masks = _runtime_masks(runtime, counts)
    active_pairs = int(sum(int(value).bit_count() for value in masks))
    widths = tuple(int(value) for value in TEMPLATE_SLOT_COUNTS)
    gtsparse_offset_rows = sum(count * width for count, width in zip(padded_counts, widths))
    full_offset_rows = _padded_rows(int(runtime.n_out), TORCHSPARSE_BM) * 27
    spconv_offset_rows = {tile_rows: _spconv_offset_rows(masks, tile_rows) for tile_rows in (32, 64, 128)}
    cin = int(features.size(1))
    cout = int(logical_weight.size(-1))
    flops_per_pair = 2 * cin * cout
    current_layers.append(
        {
            "active_pairs": active_pairs,
            "cin": cin,
            "cout": cout,
            "effective_flops": active_pairs * flops_per_pair,
            "gtsparse_issued_flops": gtsparse_offset_rows * flops_per_pair,
            "kernel_volume": 27,
            "kind": kind,
            "minkowski_issued_flops": active_pairs * flops_per_pair,
            "n_out": int(runtime.n_out),
            "padded_counts": padded_counts,
            "spconv_issued_flops_bm32": spconv_offset_rows[32] * flops_per_pair,
            "spconv_issued_flops_bm64": spconv_offset_rows[64] * flops_per_pair,
            "spconv_issued_flops_bm128": spconv_offset_rows[128] * flops_per_pair,
            "template_counts": counts,
            "torchsparse_issued_flops": full_offset_rows * flops_per_pair,
        }
    )


def _special_buffer(runtime, kind, template_id, count):
    if kind == "kernel3":
        if template_id < 3:
            return runtime.input_rows_w1[template_id, :count, :1]
        if template_id < 6:
            return runtime.input_rows_w2[template_id - 3, :count, :2]
        return runtime.input_rows_w3[0, :count, :3]
    if kind == "kernel9":
        if template_id == 0:
            return runtime.input_rows_w1[0, :count, :1]
        if template_id < 4:
            return runtime.input_rows_w4[template_id - 1, :count, : len(K9_OFFSETS[template_id])]
        if template_id < 7:
            return runtime.input_rows_w7[template_id - 4, :count, : len(K9_OFFSETS[template_id])]
        return runtime.input_rows_w9[0, :count, :9]
    if template_id == 0:
        return None
    if template_id < 9:
        return runtime.input_rows_w1[template_id - 1, :count, :1]
    if template_id < 15:
        return runtime.input_rows_w4[template_id - 9, :count, :4]
    return runtime.input_rows_w8[0, :count, :8]


def observe_special(kind, features, logical_weight, runtime, offsets) -> None:
    counts = [int(value) for value in runtime.template_counts.tolist()]
    padded_counts = [int(value) for value in runtime.padded_counts.tolist()]
    masks = []
    for template_id, count in enumerate(counts):
        if count == 0:
            continue
        template_offsets = offsets[template_id]
        if not template_offsets:
            masks.append(np.zeros(count, dtype=np.uint32))
            continue
        buffer = _special_buffer(runtime, kind, template_id, count)
        bits = torch.tensor([1 << offset for offset in template_offsets], device=buffer.device, dtype=torch.int64)
        masks.append((((buffer >= 0).to(torch.int64) * bits).sum(dim=1)).cpu().numpy().astype(np.uint32))
    masks = np.concatenate(masks) if masks else np.empty(0, dtype=np.uint32)
    active_pairs = int(sum(int(value).bit_count() for value in masks))
    issued_rows = sum(count * len(template) for count, template in zip(padded_counts, offsets))
    n_out = int(runtime.out_coords.size(0))
    kernel_volume = max(max(template, default=-1) for template in offsets) + 1
    cin = int(features.size(1))
    cout = int(logical_weight.size(-1))
    flops_per_pair = 2 * cin * cout
    spconv_rows = {tile: _spconv_offset_rows(masks, tile) for tile in (32, 64, 128)}
    current_layers.append({
        "active_pairs": active_pairs, "cin": cin, "cout": cout,
        "effective_flops": active_pairs * flops_per_pair,
        "gtsparse_issued_flops": issued_rows * flops_per_pair,
        "kernel_volume": kernel_volume, "kind": kind,
        "minkowski_issued_flops": active_pairs * flops_per_pair,
        "n_out": n_out, "padded_counts": padded_counts,
        "spconv_issued_flops_bm32": spconv_rows[32] * flops_per_pair,
        "spconv_issued_flops_bm64": spconv_rows[64] * flops_per_pair,
        "spconv_issued_flops_bm128": spconv_rows[128] * flops_per_pair,
        "template_counts": counts,
        "torchsparse_issued_flops": _padded_rows(n_out, TORCHSPARSE_BM) * kernel_volume * flops_per_pair,
    })


def special_hook(module, inputs, output):
    if not capture:
        return
    sparse_input = inputs[0]
    if isinstance(module, GeometricTemplateKernel3Conv3d):
        runtime, _ = module.build_runtime(sparse_input)
        observe_special("kernel3", sparse_input.features, module._runtime_weight(), runtime, K3_OFFSETS)
    elif isinstance(module, GeometricTemplateKernel9Conv3d):
        runtime, _ = module.build_runtime(sparse_input)
        observe_special("kernel9", sparse_input.features, module._runtime_weight(), runtime, K9_OFFSETS)
    elif isinstance(module, GeometricTemplateKernel8Conv3d):
        runtime, _ = module.build_runtime(sparse_input)
        observe_special("kernel8", sparse_input.features, module._runtime_weight(), runtime, K8_OFFSETS)
    else:
        runtime = sparse_input.metadata.reverse_chain[0].runtime.build()
        observe_special("kernel8_inverse", sparse_input.features, module._runtime_weight(), runtime, K8_OFFSETS)


def kernel27_pre_hook(module, inputs):
    global current_kernel27_kind
    if isinstance(module, GeometricTemplateSubMConv3d):
        current_kernel27_kind = "kernel27_subm"
    elif isinstance(module, GeometricTemplateSparseConv3d):
        current_kernel27_kind = "kernel27_regular"
    else:
        current_kernel27_kind = "kernel27_inverse"


def kernel27_hook(module, inputs, output):
    global current_kernel27_kind
    current_kernel27_kind = None


def observed_conv(features, logical_weight, runtime):
    if capture:
        if current_kernel27_kind is None:
            raise RuntimeError("K=27 convolution executed outside a profiled sparse module")
        observe_runtime(current_kernel27_kind, features, logical_weight, runtime)
    return original_conv(features, logical_weight, runtime)


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
    gt_ops._conv = observed_conv
    model, loader, dtype = build_workload(
        args.workload,
        "gtsparse",
        "fp16",
        args.frames,
        args.device,
        random_sample=not args.first_frames,
    )
    handles = [
        module.register_forward_hook(special_hook)
        for module in model.modules()
        if isinstance(module, (
            GeometricTemplateKernel3Conv3d,
            GeometricTemplateKernel8Conv3d,
            GeometricTemplateKernel8InverseConv3d,
            GeometricTemplateKernel9Conv3d,
        ))
    ]
    for module in model.modules():
        if isinstance(module, (
            GeometricTemplateSubMConv3d,
            GeometricTemplateSparseConv3d,
            GeometricTemplateSparseInverseConv3d,
        )):
            handles.append(module.register_forward_pre_hook(kernel27_pre_hook))
            handles.append(module.register_forward_hook(kernel27_hook))

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
            for batch in tqdm(loader, desc=f"template-profile/{args.workload}", dynamic_ncols=True):
                current_layers.clear()
                batch = move_batch(batch, args.device, dtype)
                sparse_forward(model, batch, args.workload)
                torch.cuda.synchronize(args.device)
                record = {
                    "effective_flops": sum(layer["effective_flops"] for layer in current_layers),
                    "frame_ids": list(batch.frame_ids),
                    "gpu": torch.cuda.get_device_name(torch.device(args.device)),
                    "gtsparse_issued_flops": sum(layer["gtsparse_issued_flops"] for layer in current_layers),
                    "layers": current_layers,
                    "minkowski_issued_flops": sum(layer["minkowski_issued_flops"] for layer in current_layers),
                    "torchsparse_issued_flops": sum(layer["torchsparse_issued_flops"] for layer in current_layers),
                    "workload": args.workload,
                }
                json.dump(record, output, sort_keys=True)
                output.write("\n")
                output.flush()
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    main()
