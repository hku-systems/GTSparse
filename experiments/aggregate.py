import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def insert_unique(mapping, sources, key, value, path: Path, label: str) -> None:
    if key in mapping:
        raise ValueError(f"duplicate {label} for {key}: {sources[key]} and {path}")
    mapping[key] = value
    sources[key] = path


def summary_rows(root: Path):
    rows = []
    per_frame = []
    for path in sorted(root.glob("**/*.summary.json")):
        data = json.loads(path.read_text())
        if "stats" not in data or "workload" not in data:
            continue
        relative = path.relative_to(root)
        experiment = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        min_template_match = re.search(r"min_template_(\d+)", str(relative))
        for metric, stats in data["stats"].items():
            if stats is None:
                continue
            rows.append(
                {
                    "backend": data["backend"],
                    "count": stats["count"],
                    "dtype": data["dtype"],
                    "experiment": experiment,
                    "gpu": data["gpu"],
                    "mean_ms": stats["mean_ms"],
                    "median_ms": stats["median_ms"],
                    "metric": metric,
                    "min_template": data.get("min_template", "" if min_template_match is None else min_template_match.group(1)),
                    "sweeps": data["sweeps"],
                    "workload": data["workload"],
                }
            )
        raw_path = path.with_name(path.name.replace(".summary.json", ".jsonl"))
        if experiment == "end_to_end" and raw_path.exists():
            for frame_index, record in enumerate(read_jsonl(raw_path)):
                base = {
                    "backend": data["backend"],
                    "dtype": data["dtype"],
                    "frame_id": record["frame_ids"][0],
                    "frame_index": frame_index,
                    "gpu": data["gpu"],
                    "workload": data["workload"],
                }
                if "conv_only_ms" in record:
                    per_frame.append({**base, "latency_ms": record["conv_only_ms"], "metric": "conv_only"})
                if "end2end_ms" in record:
                    per_frame.append({**base, "latency_ms": record["end2end_ms"], "metric": "end2end"})
    return rows, per_frame


def profile_rows(root: Path):
    profiles = {}
    profile_sources = {}
    raw_by_key = {}
    raw_sources = {}
    for path in sorted((root / "microbenchmark" / "profile").rglob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        key = (records[0]["gpu"], records[0]["workload"])
        insert_unique(raw_by_key, raw_sources, key, records, path, "workload profile")
    for path in sorted((root / "microbenchmark" / "template_profile").rglob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        counts = [sum(record["family_counts"][index] for record in records) for index in range(4)]
        total = sum(counts)
        if total == 0:
            raise ValueError(f"template profile contains no K=27 rows: {path}")
        widths = (1, 10, 19, 27)
        row = {
            "avg_assigned_width": sum(count * width for count, width in zip(counts, widths)) / total,
            "center_percent": 100 * counts[0] / total,
            "frames": len(records),
            "full27_percent": 100 * counts[3] / total,
            "gpu": records[0]["gpu"],
            "operator": "3x3x3",
            "skip1_percent": 100 * counts[2] / total,
            "skip2_percent": 100 * counts[1] / total,
            "workload": records[0]["workload"],
        }
        key = (row["gpu"], row["workload"])
        insert_unique(profiles, profile_sources, key, row, path, "template profile")
    spconv_by_key = {}
    spconv_sources = {}
    for path in sorted((root / "microbenchmark" / "profile_spconv").rglob("*.jsonl")):
        records = read_jsonl(path)
        if records:
            key = (records[0]["gpu"], records[0]["workload"])
            insert_unique(spconv_by_key, spconv_sources, key, records, path, "SpConv profile")
    return [profiles[key] for key in sorted(profiles)], raw_by_key, spconv_by_key


def timing_rows(root: Path):
    """Return per-frame conv-only timings keyed by backend and workload.

    Throughput is an aggregate over the same frames as the FLOP profile.  The
    timing summary's median is useful for latency tables, but pairing each
    profile record with its frame timing lets the throughput numerator and
    denominator use one consistent sample and avoids mixing mean and median
    statistics.
    """
    timings = {}
    timing_sources = {}
    timing_root = root / "microbenchmark" / "timing"
    for summary_path in sorted(timing_root.rglob("*.summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        path = summary_path.with_name(summary_path.name.replace(".summary.json", ".jsonl"))
        if not path.exists():
            continue
        records = read_jsonl(path)
        if not records:
            continue
        key = (summary["gpu"], summary["workload"], summary["backend"], summary["dtype"])
        values = {
            tuple(record["frame_ids"]): float(record["conv_only_ms"])
            for record in records
            if "conv_only_ms" in record and record.get("frame_ids")
        }
        insert_unique(timings, timing_sources, key, values, path, "microbenchmark timing")
    return timings


GTSPARSE_TEMPLATE_WIDTHS = {
    # Actual number of payload slots per template (runtime.TEMPLATE_SLOT_COUNTS),
    # i.e. what the kernel issues, not the padded family-buffer width.
    "kernel27": (1, 10, 9, 10, 18, 19, 18, 27),
    "kernel9": (1, 4, 3, 4, 6, 7, 7, 9),
    "kernel8": (0, 1, 1, 1, 1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 8),
    "kernel3": (1, 1, 1, 2, 2, 2, 3),
}
KERNEL_VOLUMES = {"kernel27": 27, "kernel9": 9, "kernel8": 8, "kernel3": 3}

# SpConv M-tiles observed from its autotune on the RTX 4090: layers with a small
# output count use M=32 and the rest use M=64.  "observed" uses this table so the
# proportionality matches the paper's RTX 4090 measurement; "autotuned" instead
# reads the M-tile SpConv autotuned on the machine that produced the logs.
SPCONV_OBSERVED_M_TILE = 64
SPCONV_OBSERVED_SMALL_M_TILE = 32
SPCONV_OBSERVED_SMALL_N_OUT = 7000
SPCONV_TILE_MODES = ("observed", "autotuned")


def sparse_kind(kind: str) -> str:
    for prefix in ("kernel27", "kernel9", "kernel8", "kernel3"):
        if kind.startswith(prefix):
            return prefix
    raise ValueError(f"unknown sparse layer kind: {kind}")


def flops_per_pair(layer) -> int:
    return 2 * int(layer["cin"]) * int(layer["cout"])


def gtsparse_issued_flops(layer) -> int:
    """Issued FLOPs over the assigned template rows, before launch tile padding."""
    widths = GTSPARSE_TEMPLATE_WIDTHS[sparse_kind(layer["kind"])]
    counts = layer["template_counts"]
    if len(counts) != len(widths):
        raise ValueError(f"template count/width mismatch for {layer['kind']}: {len(counts)} vs {len(widths)}")
    issued_rows = sum(int(count) * width for count, width in zip(counts, widths))
    return issued_rows * flops_per_pair(layer)


def full27_issued_flops(layer) -> int:
    """Issued FLOPs for a fused-offset engine that executes the full kernel."""
    volume = KERNEL_VOLUMES[sparse_kind(layer["kind"])]
    return int(layer["n_out"]) * volume * flops_per_pair(layer)


def spconv_issued_flops(layer, mode, autotuned_tile=None) -> int:
    """Issued FLOPs for SpConv's sorted skip at the selected M-tile."""
    if mode == "autotuned":
        if autotuned_tile is None:
            raise ValueError("autotuned SpConv mode needs a tile from the SpConv profile")
        tile = int(autotuned_tile)
    else:
        tile = (
            SPCONV_OBSERVED_SMALL_M_TILE
            if int(layer["n_out"]) < SPCONV_OBSERVED_SMALL_N_OUT
            else SPCONV_OBSERVED_M_TILE
        )
    return int(layer[f"spconv_issued_flops_bm{tile}"])


def throughput_rows(summary, raw_profiles, spconv_profiles, frame_timings=None, spconv_tile_mode="observed"):
    rows = []
    for timing in summary:
        if timing["experiment"] != "microbenchmark" or timing["metric"] != "conv_only":
            continue
        key = (timing["gpu"], timing["workload"])
        records = raw_profiles.get(key)
        if not records:
            continue
        backend = timing["backend"]
        spconv_by_frame = None
        if spconv_tile_mode == "autotuned":
            spconv_records = spconv_profiles.get(key)
            if not spconv_records:
                raise ValueError(f"autotuned SpConv mode needs a SpConv profile for {key}")
            spconv_by_frame = {tuple(record["frame_ids"]): record for record in spconv_records}
        timing_by_frame = None
        if frame_timings is not None:
            timing_by_frame = frame_timings.get((timing["gpu"], timing["workload"], backend, timing["dtype"]))
            if not timing_by_frame:
                continue
            missing = [tuple(record["frame_ids"]) for record in records if tuple(record["frame_ids"]) not in timing_by_frame]
            if missing:
                raise ValueError(
                    f"throughput profile/timing frame mismatch for {timing['gpu']} "
                    f"{timing['workload']} {backend}: {len(missing)} profile frames have no timing"
                )
            elapsed_ms = sum(timing_by_frame[tuple(record["frame_ids"])] for record in records)
            effective = sum(record["effective_flops"] for record in records)
        else:
            # Backward-compatible path for callers that only have summaries.
            elapsed_ms = float(timing["median_ms"]) * len(records)
            effective = statistics.mean(record["effective_flops"] for record in records) * len(records)
        if backend == "gtsparse":
            issued = sum(gtsparse_issued_flops(layer) for record in records for layer in record["layers"])
        elif backend == "spconv":
            issued = 0
            for record in records:
                spconv_record = spconv_by_frame.get(tuple(record["frame_ids"])) if spconv_by_frame else None
                for index, layer in enumerate(record["layers"]):
                    tile = spconv_record["layers"][index]["tile_rows"] if spconv_record else None
                    issued += spconv_issued_flops(layer, spconv_tile_mode, tile)
        elif backend == "torchsparse":
            issued = 0
            for record in records:
                spconv_record = spconv_by_frame.get(tuple(record["frame_ids"])) if spconv_by_frame else None
                bev_seen = False
                for index, layer in enumerate(record["layers"]):
                    if (
                        record["workload"].startswith("voxelnext_")
                        and sparse_kind(layer["kind"]) == "kernel9"
                        and not bev_seen
                    ):
                        # VoxelNeXt's BEV tail is a SpConv convolution.
                        tile = spconv_record["layers"][index]["tile_rows"] if spconv_record else None
                        issued += spconv_issued_flops(layer, spconv_tile_mode, tile)
                        bev_seen = True
                    else:
                        issued += full27_issued_flops(layer)
        elif backend == "minkowski":
            issued = sum(record["minkowski_issued_flops"] for record in records)
        else:
            continue
        rows.append(
            {
                "backend": backend,
                "dtype": timing["dtype"],
                "effective_tflops": effective / elapsed_ms / 1e9,
                "gpu": timing["gpu"],
                "proportionality_percent": 100 * effective / issued,
                "raw_tflops": issued / elapsed_ms / 1e9,
                "workload": timing["workload"],
            }
        )
    rows.sort(key=lambda row: (row["gpu"], row["workload"], row["backend"], row["dtype"]))
    return rows


def breakdown_rows(root: Path, summaries):
    rows = []
    seen = {}
    for path in sorted((root / "microbenchmark" / "time_breakdown").rglob("*.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        backend = records[0]["backend"]
        dtype = records[0]["dtype"]
        gpu = records[0]["gpu"]
        workload = records[0]["workload"]
        key = (gpu, workload, backend, dtype)
        if key in seen:
            raise ValueError(f"duplicate time breakdown for {key}: {seen[key]} and {path}")
        seen[key] = path
        metric = "end2end" if workload.startswith("minkunet") else "conv_only"
        targets = [
            row
            for row in summaries
            if row["experiment"] == "end_to_end"
            and row["gpu"] == gpu
            and row["workload"] == workload
            and row["backend"] == backend
            and row["dtype"] == dtype
            and row["metric"] == metric
        ]
        scale_experiment = "end_to_end"
        scale_metric = metric
        if not targets:
            targets = [
                row
                for row in summaries
                if row["experiment"] == "microbenchmark"
                and row["gpu"] == gpu
                and row["workload"] == workload
                and row["backend"] == backend
                and row["dtype"] == dtype
                and row["metric"] == metric
            ]
            scale_experiment = "microbenchmark"
        if targets:
            target_ms = float(targets[-1]["median_ms"])
        else:
            # Standalone microbenchmark: no end-to-end or timing summary to scale
            # to, so fall back to the measured breakdown total.
            target_ms = statistics.median(record["total_ms"] for record in records)
            scale_experiment = "raw"
            scale_metric = "total"
        raw_builder = statistics.median(record["builder_ms"] for record in records)
        raw_kernel = statistics.median(record["kernel_ms"] for record in records)
        builder_share = statistics.median(
            record["builder_ms"] / record["total_ms"] for record in records
        )
        rows.append(
            {
                "backend": backend,
                "builder_median_ms": builder_share * target_ms,
                "builder_percent": 100 * builder_share,
                "dtype": dtype,
                "frames": len(records),
                "gpu": gpu,
                "kernel_median_ms": (1 - builder_share) * target_ms,
                "kernel_percent": 100 * (1 - builder_share),
                "method": records[0]["method"],
                "raw_builder_median_ms": raw_builder,
                "raw_kernel_median_ms": raw_kernel,
                "scale_experiment": scale_experiment,
                "scale_metric": scale_metric,
                "total_median_ms": target_ms,
                "workload": workload,
            }
        )
    rows.sort(key=lambda row: (row["gpu"], row["workload"], row["backend"], row["dtype"]))
    return rows


def memory_rows(root: Path):
    rows = {}
    sources = {}
    for path in sorted((root / "microbenchmark" / "peak_memory").rglob("*.json")):
        row = json.loads(path.read_text())
        key = (row["gpu"], row["workload"], row["backend"], row["dtype"])
        insert_unique(rows, sources, key, row, path, "peak-memory record")
    return [rows[key] for key in sorted(rows)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--spconv-tile",
        choices=SPCONV_TILE_MODES,
        default="observed",
        help="SpConv M-tile: 'observed' uses the RTX 4090 autotune table, "
        "'autotuned' reads the tile autotuned on this machine",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries, per_frame = summary_rows(args.logs)
    profiles, raw_profiles, spconv_profiles = profile_rows(args.logs)
    frame_timings = timing_rows(args.logs)
    throughput = throughput_rows(summaries, raw_profiles, spconv_profiles, frame_timings, args.spconv_tile)
    breakdown = breakdown_rows(args.logs, summaries)
    memory = memory_rows(args.logs)

    write_csv(args.results / "latency.csv", ("experiment", "gpu", "dtype", "workload", "sweeps", "backend", "metric", "count", "median_ms", "mean_ms", "min_template"), summaries)
    write_csv(args.results / "per_frame_latency.csv", ("gpu", "dtype", "workload", "backend", "metric", "frame_index", "frame_id", "latency_ms"), per_frame)
    write_csv(args.results / "template_distribution.csv", ("gpu", "workload", "operator", "frames", "center_percent", "skip2_percent", "skip1_percent", "full27_percent", "avg_assigned_width"), profiles)
    write_csv(args.results / "effective_throughput.csv", ("gpu", "workload", "backend", "dtype", "raw_tflops", "effective_tflops", "proportionality_percent"), throughput)
    write_csv(
        args.results / "time_breakdown.csv",
        (
            "gpu",
            "workload",
            "backend",
            "dtype",
            "frames",
            "method",
            "scale_experiment",
            "scale_metric",
            "builder_percent",
            "kernel_percent",
            "raw_builder_median_ms",
            "raw_kernel_median_ms",
            "builder_median_ms",
            "kernel_median_ms",
            "total_median_ms",
        ),
        breakdown,
    )
    write_csv(args.results / "peak_memory.csv", ("gpu", "workload", "backend", "dtype", "frames", "peak_memory_mb"), memory)


if __name__ == "__main__":
    main()
