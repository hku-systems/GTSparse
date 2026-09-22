import argparse
import csv
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import Patch, Rectangle


BACKENDS = ("minkowski", "spconv", "torchsparse", "gtsparse")
LABELS = {"minkowski": "Minkowski", "spconv": "SpConv", "torchsparse": "TS++", "gtsparse": "GTSparse"}
COLORS = {"minkowski": "#b3cde3", "spconv": "#ccebc5", "torchsparse": "#fddaec", "gtsparse": "#fed9a6"}
HATCHES = {"minkowski": "//", "spconv": "\\\\", "torchsparse": "xx", "gtsparse": ""}
LINE_COLORS = {"minkowski": "#3a7cb8", "spconv": "#2d8e2d", "torchsparse": "#c4386b", "gtsparse": "#d4760a"}
MARKERS = {"minkowski": "s", "spconv": "^", "torchsparse": "x", "gtsparse": "o"}
WORKLOAD_LABELS = {
    "second_kitti_sweeps1": "SECOND",
    "voxelnext_nuscenes_sweeps1": "VoxelNeXt (sw=1)",
    "voxelnext_nuscenes_sweeps10": "VoxelNeXt (sw=10)",
    "minkunet_semantickitti_sweeps1": "MinkUNet42",
}
PAPER_WORKLOAD_LABELS = {
    "second_kitti_sweeps1": "SECOND",
    "voxelnext_nuscenes_sweeps1": "VoxelNeXt\n(sw=1)",
    "voxelnext_nuscenes_sweeps10": "VoxelNeXt\n(sw=10)",
    "minkunet_semantickitti_sweeps1": "MinkUNet",
}
PAPER_PANEL_LABELS = {
    "RTX 4090": "(a)",
    "A100": "(b)",
    "RTX 4070 Laptop": "(c)",
    "RTX 3080": "(d)",
    "AGX Orin": "(e)",
    "RTX 2080 Ti": "(f)",
}
PANEL_GPU_LABELS = {
    "RTX 4090": "4090",
    "A100": "A100",
    "RTX 4070 Laptop": "4070 Laptop",
    "RTX 3080": "3080",
    "AGX Orin": "AGX Orin",
    "RTX 2080 Ti": "2080 Ti",
}


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as source:
        return list(csv.DictReader(source))


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def short_gpu_name(gpu: str) -> str:
    name = gpu.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
    if "4070" in name and "Laptop" in name:
        return "RTX 4070 Laptop"
    if "4090" in name:
        return "RTX 4090"
    if "3080" in name:
        return "RTX 3080"
    if "2080" in name and "Ti" in name:
        return "RTX 2080 Ti"
    if "A100" in name:
        return "A100"
    if "Orin" in name:
        return "AGX Orin"
    return name.replace("_", " ")


def single_gpu(rows) -> str:
    """Return the only platform in a single-platform figure's input rows."""
    gpus = sorted({row["gpu"] for row in rows if row.get("gpu")})
    if not gpus:
        return ""
    if len(gpus) != 1:
        raise ValueError(f"single-platform plot received multiple GPUs: {gpus}")
    return gpus[0]


def gpu_path_name(gpu: str) -> str:
    return "".join(character if character.isalnum() or character in "-_." else "_" for character in gpu).strip("_") or "unknown_gpu"


def gpu_sort_key(gpu: str):
    paper_order = {name: index for index, name in enumerate(PAPER_PANEL_LABELS)}
    return paper_order.get(short_gpu_name(gpu), len(paper_order)), short_gpu_name(gpu)


def finite_value(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def caption_text(fig, label: str, caption: str, *, x: float, y: float, width: int, fontsize: float, location: str) -> None:
    """Add a paper-style numbered caption without changing the plot axes."""
    numbered_caption(fig, label, caption, x=x, y=y, width=width, fontsize=fontsize, location=location)


def numbered_caption(fig, label: str, caption: str, *, x: float, y: float, width: int, fontsize: float, location: str) -> None:
    lines = []
    for paragraph in caption.splitlines():
        lines.extend(textwrap.wrap(paragraph, width=width, break_long_words=False, break_on_hyphens=False) or [""])
    first_line = lines[0] if lines else ""
    first = HPacker(
        children=(
            TextArea(label, textprops={"fontsize": fontsize, "fontfamily": "serif", "fontweight": "bold"}),
            TextArea(" " + first_line, textprops={"fontsize": fontsize, "fontfamily": "serif"}),
        ),
        align="baseline",
        pad=0,
        sep=0,
    )
    packed = VPacker(
        children=(first, *(TextArea(line, textprops={"fontsize": fontsize, "fontfamily": "serif"}) for line in lines[1:])),
        align="left",
        pad=0,
        sep=1,
    )
    fig.add_artist(
        AnchoredOffsetbox(
            loc=location,
            child=packed,
            frameon=False,
            pad=0,
            borderpad=0,
            bbox_to_anchor=(x, y),
            bbox_transform=fig.transFigure,
        )
    )


def table_canvas(column_widths, row_count: int, header_rows: int, figure_width: float, caption_label: str, caption: str):
    total_width = float(sum(column_widths))
    table_height = row_count + header_rows
    caption_rows = 1.35
    fig, axis = plt.subplots(figsize=(figure_width, 0.36 * table_height + 0.72))
    axis.axis("off")
    axis.set_xlim(0.0, total_width)
    axis.set_ylim(0.0, float(table_height) + caption_rows)
    edges = np.concatenate(([0.0], np.cumsum(column_widths)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    numbered_caption(
        fig,
        caption_label,
        caption,
        x=0.04,
        y=0.96,
        width=max(68, int(figure_width * 14)),
        fontsize=9.5,
        location="upper left",
    )
    return fig, axis, edges, centers


def table_rule(axis, y: float, start: float, end: float, linewidth: float) -> None:
    axis.plot((start, end), (y, y), color="black", linewidth=linewidth, clip_on=False)


def plot_end_to_end(rows, figures: Path, dtype: str) -> None:
    rows = [row for row in rows if row["experiment"] == "end_to_end"]
    gpus = sorted({row["gpu"] for row in rows if row["backend"] == "gtsparse" and row["dtype"] == dtype})
    workloads = [name for name in PAPER_WORKLOAD_LABELS if any(row["workload"] == name for row in rows)]
    if not gpus or not workloads:
        return

    gpus.sort(key=gpu_sort_key)
    n_workloads = len(workloads)
    n_systems = len(BACKENDS)
    columns = 1 if len(gpus) == 1 else 2
    rows_count = math.ceil(len(gpus) / columns)

    sub_width = 0.78
    sub_height = 1.25
    workload_gap = 0.28
    panel_gap_x = 0.25
    panel_gap_y = 0.28
    gpu_label_width = 0.50
    margin_left = 0.05
    margin_right = 0.05
    margin_top = 0.42
    margin_bottom = 0.65
    caption_height = 0.78
    panel_width = n_workloads * sub_width + (n_workloads - 1) * workload_gap
    figure_width = margin_left + columns * (gpu_label_width + panel_width) + (columns - 1) * panel_gap_x + margin_right
    figure_height = margin_top + rows_count * sub_height + (rows_count - 1) * panel_gap_y + margin_bottom + caption_height
    figure = plt.figure(figsize=(figure_width, figure_height))
    bar_width = 0.62

    for gpu_index, gpu in enumerate(gpus):
        row_index = gpu_index // columns
        column_index = gpu_index % columns
        gpu_label = short_gpu_name(gpu)
        panel_gpu_label = PANEL_GPU_LABELS.get(gpu_label, gpu_label)
        x_origin = margin_left + gpu_label_width + column_index * (gpu_label_width + panel_width + panel_gap_x)
        y_origin = figure_height - margin_top - (row_index + 1) * sub_height - row_index * panel_gap_y
        bottom_row = row_index == rows_count - 1

        for workload_index, workload in enumerate(workloads):
            left = (x_origin + workload_index * (sub_width + workload_gap)) / figure_width
            bottom = y_origin / figure_height
            width = sub_width / figure_width
            height = sub_height / figure_height
            metric = "end2end" if workload.startswith("minkunet") else "conv_only"
            values = []
            for backend in BACKENDS:
                actual_dtype = "fp32" if backend == "minkowski" else dtype
                match = [row for row in rows if row["gpu"] == gpu and row["workload"] == workload and row["backend"] == backend and row["dtype"] == actual_dtype and row["metric"] == metric]
                values.append(float(match[-1]["median_ms"]) if match else np.nan)

            broken_platforms = {"RTX 4070 Laptop", "RTX 3080", "AGX Orin"}
            finite_minkowski = finite_value(values[BACKENDS.index("minkowski")])
            finite_other = [finite_value(value) for backend, value in zip(BACKENDS, values) if backend != "minkowski"]
            finite_other = [value for value in finite_other if value is not None]
            broken = (
                workload == "minkunet_semantickitti_sweeps1"
                and gpu_label in broken_platforms
                and finite_minkowski is not None
                and finite_other
                and finite_minkowski > max(finite_other) * 1.35
            )
            if broken:
                minkowski_value = finite_minkowski
                lower_values = finite_other
                lower_top = max(lower_values) * 1.35
                upper_bottom = minkowski_value * 0.92
                upper_top = minkowski_value * 1.10
                lower_ratio = 0.72
                gap_ratio = 0.06
                upper_ratio = 1.0 - lower_ratio - gap_ratio
                lower_axis = figure.add_axes((left, bottom, width, height * lower_ratio))
                upper_axis = figure.add_axes((left, bottom + height * (lower_ratio + gap_ratio), width, height * upper_ratio))

                for axis in (lower_axis, upper_axis):
                    for index, (backend, value) in enumerate(zip(BACKENDS, values)):
                        axis.bar(index, value, bar_width, color=COLORS[backend], hatch=HATCHES[backend], edgecolor="black", linewidth=0.5, zorder=3)
                    axis.set_xlim(-0.5, n_systems - 0.5)
                    axis.set_xticks([])
                    axis.set_axisbelow(True)
                    axis.yaxis.grid(True, alpha=0.25, linewidth=0.4)

                lower_axis.set_ylim(0, lower_top)
                lower_axis.tick_params(axis="y", labelsize=9, pad=1, length=2)
                lower_axis.spines["top"].set_visible(False)
                lower_axis.spines["right"].set_visible(False)
                upper_axis.set_ylim(upper_bottom, upper_top)
                upper_axis.set_yticks([round(minkowski_value)])
                upper_axis.tick_params(axis="y", labelsize=9, pad=1, length=2, bottom=False)
                upper_axis.spines["bottom"].set_visible(False)
                upper_axis.spines["right"].set_visible(False)
                upper_axis.spines["top"].set_visible(False)

                break_length = 0.004
                break_delta_y = break_length * np.tan(np.radians(30))
                break_style = dict(color="black", clip_on=False, linewidth=0.8, transform=figure.transFigure)
                break_x = left
                lower_edge = bottom + height * lower_ratio
                upper_edge = bottom + height * (lower_ratio + gap_ratio)
                figure.lines.extend(
                    (
                        plt.Line2D((break_x - break_length, break_x + break_length), (lower_edge - break_delta_y, lower_edge + break_delta_y), **break_style),
                        plt.Line2D((break_x - break_length, break_x + break_length), (upper_edge - break_delta_y, upper_edge + break_delta_y), **break_style),
                    )
                )

                minkowski_index = BACKENDS.index("minkowski")
                wave_x = np.linspace(minkowski_index - bar_width / 2, minkowski_index + bar_width / 2, 80)
                wave_amplitude = lower_top * 0.025
                lower_wave = lower_top - wave_amplitude + wave_amplitude * np.sin((wave_x - wave_x[0]) / bar_width * 2 * np.pi)
                lower_axis.fill_between(wave_x, lower_wave, lower_top * 1.1, color="white", zorder=6, clip_on=True)
                lower_axis.plot(wave_x, lower_wave, color="black", linewidth=0.5, zorder=7, clip_on=True)
                visible_upper_bottom = upper_bottom + (upper_top - upper_bottom) * 0.15
                upper_wave = visible_upper_bottom - wave_amplitude + wave_amplitude * np.sin((wave_x - wave_x[0]) / bar_width * 2 * np.pi)
                upper_axis.fill_between(wave_x, upper_bottom * 0.9, upper_wave, color="white", zorder=6, clip_on=True)
                upper_axis.plot(wave_x, upper_wave, color="black", linewidth=0.5, zorder=7, clip_on=True)

                if workload_index == 0:
                    lower_axis.set_ylabel("Latency (ms)", fontsize=9.5)
                    lower_axis.yaxis.set_label_coords(-0.25, 0.65)
                if bottom_row:
                    lower_axis.set_xlabel(PAPER_WORKLOAD_LABELS[workload], fontsize=9.5, labelpad=8)
            else:
                axis = figure.add_axes((left, bottom, width, height))
                for index, (backend, value) in enumerate(zip(BACKENDS, values)):
                    axis.bar(index, value, bar_width, color=COLORS[backend], hatch=HATCHES[backend], edgecolor="black", linewidth=0.5, zorder=3)
                axis.set_xlim(-0.5, n_systems - 0.5)
                axis.set_xticks([])
                axis.tick_params(axis="y", labelsize=9, pad=1, length=2)
                axis.set_axisbelow(True)
                axis.yaxis.grid(True, alpha=0.25, linewidth=0.4)
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
                if workload_index == 0:
                    axis.set_ylabel("Latency (ms)", fontsize=9.5)
                    axis.yaxis.set_label_coords(-0.25, 0.5)
                if bottom_row:
                    axis.set_xlabel(PAPER_WORKLOAD_LABELS[workload], fontsize=9.5, labelpad=8)

        panel_letter = PAPER_PANEL_LABELS.get(gpu_label, f"({chr(ord('a') + gpu_index)})")
        figure.text(
            (x_origin - gpu_label_width * 0.85) / figure_width,
            (y_origin + sub_height * 0.5) / figure_height,
            f"{panel_letter} {panel_gpu_label} ({dtype.upper()})",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            rotation=90,
        )
        panel_box = Rectangle(
            ((x_origin - 0.35) / figure_width, (y_origin - 0.08) / figure_height),
            (panel_width + 0.35 + 0.06) / figure_width,
            (sub_height + 0.16) / figure_height,
            transform=figure.transFigure,
            linewidth=0.5,
            edgecolor="black",
            facecolor="none",
            clip_on=False,
            zorder=0,
        )
        figure.patches.append(panel_box)

    handles = [Patch(facecolor=COLORS[name], hatch=HATCHES[name], edgecolor="black", linewidth=0.5, label=LABELS[name]) for name in BACKENDS]
    legend = figure.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=10,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.5,
    )
    for text in legend.get_texts():
        if text.get_text() == "GTSparse":
            text.set_fontweight("bold")
    if len(gpus) == 1:
        platform_phrase = f"on {short_gpu_name(gpus[0])}"
    else:
        platform_phrase = f"across {len(gpus)} GPU platforms ({', '.join(short_gpu_name(gpu) for gpu in gpus)})"
    caption_text(
        figure,
        "Figure 4.",
        f"End-to-end inference latency {platform_phrase} (FP16 unless noted). Each panel shows one GPU with four workload configurations, each plotted with an independent y-axis (latency in ms) due to differing latency ranges. Within each workload, four bars compare Minkowski, SpConv, TS++, and GTSparse. Detection workloads report sparse convolution latency; segmentation reports full pipeline latency. Lower is better.",
        x=0.035,
        y=0.035,
        width=max(72, int(figure_width * 17)),
        fontsize=7.5,
        location="lower left",
    )
    save(figure, figures / f"end_to_end_{dtype}.pdf")


def plot_template_distribution(rows, figures: Path) -> None:
    if not rows:
        return
    platform = single_gpu(rows)
    if platform:
        rows = [row for row in rows if row.get("gpu") == platform]
    order = {name: index for index, name in enumerate(WORKLOAD_LABELS)}
    rows.sort(key=lambda row: order.get(row["workload"], len(order)))
    widths = (2.15, 1.05, 1.20, 1.20, 1.05, 1.15)
    fig, axis, edges, centers = table_canvas(
        widths,
        len(rows),
        2,
        7.2,
        "Table 2.",
        "Template assignment distribution (% of output voxels per family). The template set is constructed from KITTI/SECOND but generalizes across datasets.",
    )
    top = len(rows) + 2
    table_rule(axis, top, edges[0], edges[-1], 1.2)
    table_rule(axis, len(rows) + 1, edges[1] + 0.06, edges[5] - 0.06, 0.8)
    table_rule(axis, len(rows), edges[0], edges[-1], 0.8)
    table_rule(axis, 0, edges[0], edges[-1], 1.2)
    axis.text(centers[0], len(rows) + 1, "Workload", ha="center", va="center", fontsize=10, fontweight="bold")
    axis.text((edges[1] + edges[5]) / 2, len(rows) + 1.5, "Template Family (%)", ha="center", va="center", fontsize=10, fontweight="bold")
    axis.text(centers[5], len(rows) + 1, "Avg\nWidth", ha="center", va="center", fontsize=10, fontweight="bold", linespacing=1.15)
    for column, label in enumerate(("center", "skip(2,3)", "skip(1,3)", "full27"), start=1):
        axis.text(centers[column], len(rows) + 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold")
    for row_index, row in enumerate(rows):
        y = len(rows) - row_index - 0.5
        values = (
            WORKLOAD_LABELS.get(row["workload"], row["workload"]),
            f"{float(row['center_percent']):.1f}",
            f"{float(row['skip2_percent']):.1f}",
            f"{float(row['skip1_percent']):.1f}",
            f"{float(row['full27_percent']):.1f}",
            f"{float(row['avg_assigned_width']):.1f}",
        )
        for column, value in enumerate(values):
            axis.text(centers[column], y, value, ha="center", va="center", fontsize=10)
    save(fig, figures / "template_distribution.pdf")


def plot_effective_throughput(rows, figures: Path) -> None:
    if not rows:
        return
    platform = single_gpu(rows)
    if platform:
        rows = [row for row in rows if row.get("gpu") == platform]
    workloads = sorted({row["workload"] for row in rows})
    for workload in workloads:
        selected = [row for row in rows if row["workload"] == workload]
        values = {row["backend"]: row for row in selected}
        if not values:
            continue
        group_x = np.array([0.0, 1.6])
        width = 0.18
        fig, axis = plt.subplots(figsize=(5.5, 2.8))
        for index, backend in enumerate(BACKENDS):
            if backend not in values:
                continue
            offset = (index - 1.5) * (width + 0.04)
            axis.bar(
                group_x + offset,
                [float(values[backend]["raw_tflops"]), float(values[backend]["effective_tflops"])],
                width,
                label=LABELS[backend],
                color=COLORS[backend],
                hatch=HATCHES[backend],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )
        axis.set_xticks(group_x, ("Raw", "Effective"))
        axis.set_ylabel("Throughput (TFLOPS)", fontsize=12)
        axis.tick_params(labelsize=11)
        axis.set_xlim(group_x[0] - 0.6, group_x[1] + 0.6)
        axis.set_ylim(0, max(float(row["raw_tflops"]) for row in selected) * 1.15)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.25, linewidth=0.3)
        legend = axis.legend(framealpha=0.8, fontsize=11, ncol=2, loc="upper right", handlelength=1.5, columnspacing=1.0)
        for text in legend.get_texts():
            if text.get_text() == "GTSparse":
                text.set_fontweight("bold")
        fig.tight_layout()
        platform_label = short_gpu_name(platform) if platform else "the measured GPU"
        selected_dtypes = {
            row.get("dtype")
            for row in selected
            if row.get("dtype") and row.get("backend") != "minkowski"
        }
        if not selected_dtypes:
            selected_dtypes = {row.get("dtype") for row in selected if row.get("dtype")}
        precision = next(iter(selected_dtypes), "fp16").upper()
        if workload.startswith("voxelnext_"):
            label = f"VoxelNeXt/nuScenes (sw={workload.rsplit('sweeps', 1)[-1]}, {precision}, {platform_label})"
        else:
            label = f"{WORKLOAD_LABELS.get(workload, workload)} ({precision}, {platform_label})"
        ts = values.get("torchsparse")
        sp = values.get("spconv")
        gt = values.get("gtsparse")
        if ts and sp and gt:
            ts_raw = float(ts["raw_tflops"])
            sp_raw = float(sp["raw_tflops"])
            gt_raw = float(gt["raw_tflops"])
            ts_prop = float(ts["proportionality_percent"])
            sp_prop = float(sp["proportionality_percent"])
            gt_prop = float(gt["proportionality_percent"])
            caption = (
                f"Raw and effective computation throughput on {label}. "
                f"TS++ and GTSparse achieve comparable raw throughput ({ts_raw:.1f} and {gt_raw:.1f} TFLOPS), but TS++ converts only {ts_prop:.1f}% into useful work. "
                f"SpConv's sort+skip reduces issued computation but its preprocessing overhead limits raw throughput to {sp_raw:.1f} TFLOPS. "
                f"GTSparse achieves {gt_prop:.1f}% proportionality, comparable to SpConv's {sp_prop:.1f}%, at {gt_raw / sp_raw:.1f}x the raw throughput."
            )
        else:
            caption = f"Raw and effective computation throughput on {label}."
        caption_text(
            fig,
            "Figure 5.",
            caption,
            x=0.04,
            y=-0.25,
            width=82,
            fontsize=7.5,
            location="lower left",
        )
        save(fig, figures / f"effective_throughput_{workload}.pdf")


def plot_time_breakdown(rows, figures: Path) -> None:
    if not rows:
        return
    platform = single_gpu(rows)
    if platform:
        rows = [row for row in rows if row.get("gpu") == platform]
    workload_order = {name: index for index, name in enumerate(WORKLOAD_LABELS)}
    backend_order = {name: index for index, name in enumerate(BACKENDS)}
    rows.sort(key=lambda row: (workload_order.get(row["workload"], len(workload_order)), backend_order[row["backend"]]))
    platform = short_gpu_name(platform) if platform else "the measured GPU"
    widths = (2.25, 1.45, 1.25, 1.25, 1.15)
    fig, axis, edges, centers = table_canvas(
        widths,
        len(rows),
        1,
        6.4,
        "Table 3.",
        f"Time breakdown (ms) on {platform}, FP16. Builder time measures runtime construction (coordinate hashing, indice computation, or template classification); kernel time measures sparse convolution execution.",
    )
    table_rule(axis, len(rows) + 1, edges[0], edges[-1], 1.2)
    table_rule(axis, len(rows), edges[0], edges[-1], 0.8)
    table_rule(axis, 0, edges[0], edges[-1], 1.2)
    for column, label in enumerate(("Workload", "System", "Builder\n(ms)", "Kernel\n(ms)", "Total\n(ms)")):
        axis.text(centers[column], len(rows) + 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold", linespacing=1.1)
    grouped = []
    for workload in sorted({row["workload"] for row in rows}, key=lambda name: workload_order.get(name, len(workload_order))):
        grouped.append((workload, [row for row in rows if row["workload"] == workload]))
    cursor = 0
    for workload, workload_rows in grouped:
        if cursor:
            table_rule(axis, len(rows) - cursor, edges[0], edges[-1], 0.8)
        group_center = len(rows) - cursor - len(workload_rows) / 2
        axis.text(centers[0], group_center, WORKLOAD_LABELS.get(workload, workload), ha="center", va="center", fontsize=10)
        for row in workload_rows:
            y = len(rows) - cursor - 0.5
            values = (
                LABELS[row["backend"]],
                f"{float(row['builder_median_ms']):.2f}",
                f"{float(row['kernel_median_ms']):.2f}",
                f"{float(row['total_median_ms']):.2f}",
            )
            weight = "bold" if row["backend"] == "gtsparse" else "normal"
            for column, value in enumerate(values, start=1):
                axis.text(centers[column], y, value, ha="center", va="center", fontsize=10, fontweight=weight)
            cursor += 1
    save(fig, figures / "time_breakdown.pdf")


def plot_peak_memory(rows, figures: Path) -> None:
    platform = single_gpu(rows)
    if platform:
        rows = [row for row in rows if row.get("gpu") == platform]
    workloads = sorted({row["workload"] for row in rows})
    if not workloads:
        return
    workload_order = {name: index for index, name in enumerate(WORKLOAD_LABELS)}
    workloads.sort(key=lambda workload: workload_order.get(workload, len(workload_order)))
    platform = short_gpu_name(platform) if platform else "the measured GPU"
    widths = (2.15, 1.35, 1.20, 1.05, 1.35)
    fig, axis, edges, centers = table_canvas(
        widths,
        len(workloads),
        1,
        6.8,
        "Table 4.",
        f"Peak GPU memory usage (MB) on {platform}, FP16. GTSparse's worst-case pre-allocation incurs modest additional memory compared to baselines.",
    )
    table_rule(axis, len(workloads) + 1, edges[0], edges[-1], 1.2)
    table_rule(axis, len(workloads), edges[0], edges[-1], 0.8)
    table_rule(axis, 0, edges[0], edges[-1], 1.2)
    for column, label in enumerate(("Workload", "Minkowski", "SpConv", "TS++", "GTSparse")):
        axis.text(centers[column], len(workloads) + 0.5, label, ha="center", va="center", fontsize=10, fontweight="bold")
    for row_index, workload in enumerate(workloads):
        y = len(workloads) - row_index - 0.5
        selected = {row["backend"]: row for row in rows if row["workload"] == workload}
        axis.text(centers[0], y, WORKLOAD_LABELS.get(workload, workload), ha="center", va="center", fontsize=10)
        for column, backend in enumerate(BACKENDS, start=1):
            value = f"{float(selected[backend]['peak_memory_mb']):.1f}" if backend in selected else "--"
            weight = "bold" if backend == "gtsparse" else "normal"
            axis.text(centers[column], y, value, ha="center", va="center", fontsize=10, fontweight=weight)
    save(fig, figures / "peak_memory.pdf")


def plot_ablation(rows, figures: Path) -> None:
    rows = [row for row in rows if row["experiment"] == "ablation" and row["backend"] == "gtsparse" and row["metric"] == "conv_only"]
    if not rows:
        return
    platform = single_gpu(rows)
    if platform:
        rows = [row for row in rows if row.get("gpu") == platform]
    rows.sort(key=lambda row: int(row["min_template"]))
    labels = {
        "0": "All templates",
        "1": "No center",
        "4": "No center\n+ no skip(2,3)",
        "7": "Full27 only",
    }
    colors = ("#fed9a6", "#f9c67a", "#f4a84e", "#d4760a")
    hatches = ("", "//", "\\", "xx")
    values = [float(row["median_ms"]) for row in rows]
    fig, axis = plt.subplots(figsize=(3.2, 3.0))
    for index, value in enumerate(values):
        axis.bar(index, value, 0.6, color=colors[index], hatch=hatches[index], edgecolor="black", linewidth=0.6)
    axis.set_xticks([])
    axis.set_ylabel("Latency (ms)", fontsize=13)
    axis.tick_params(axis="y", labelsize=12)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.25, linewidth=0.3)
    ymin = min(values) * 0.88
    ymax = max(values) * 1.02
    axis.set_ylim(ymin, ymax)
    for index, value in enumerate(values):
        axis.text(index, value + (ymax - ymin) * 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=11)
    handles = [Patch(facecolor=colors[index], hatch=hatches[index], edgecolor="black", linewidth=0.6, label=labels[row["min_template"]]) for index, row in enumerate(rows)]
    axis.legend(handles=handles, fontsize=11, loc="lower left", framealpha=0.9, handlelength=1.0, ncol=2, bbox_to_anchor=(-0.3, 1.02), columnspacing=0.6, handletextpad=0.4)
    fig.tight_layout()
    platform_label = short_gpu_name(platform) if platform else "the measured GPU"
    caption_text(
        fig,
        "Figure 7(a).",
        f"Effect of progressively removing template families (VoxelNeXt/nuScenes, sw=10, FP16, {platform_label}).",
        x=0.04,
        y=-0.25,
        width=64,
        fontsize=7.5,
        location="lower left",
    )
    save(fig, figures / "ablation.pdf")


def plot_sensitivity(rows, figures: Path) -> None:
    rows = [row for row in rows if row["experiment"] == "sensitivity" and row["metric"] == "conv_only"]
    if not rows:
        return
    gpu = single_gpu(rows)
    rows = [row for row in rows if row["gpu"] == gpu]
    sweeps = sorted({int(row["sweeps"]) for row in rows})
    fig, axis = plt.subplots(figsize=(2.4, 2.4))
    for backend in BACKENDS:
        selected = sorted((row for row in rows if row["backend"] == backend), key=lambda row: int(row["sweeps"]))
        if selected:
            axis.plot(
                range(len(selected)),
                [float(row["median_ms"]) for row in selected],
                marker=MARKERS[backend],
                color=LINE_COLORS[backend],
                label=LABELS[backend],
                linewidth=1.5,
                markersize=6,
                markeredgewidth=1.2,
                markeredgecolor="black",
                markerfacecolor=LINE_COLORS[backend],
            )
    axis.set_xticks(range(len(sweeps)), [str(value) for value in sweeps], fontsize=11)
    if len(sweeps) > 1:
        axis.set_xlim(-0.3, len(sweeps) - 0.7)
    axis.set_xlabel("Number of fused sweeps", fontsize=11)
    axis.set_ylabel("Latency (ms)", fontsize=11)
    axis.tick_params(axis="y", labelsize=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(alpha=0.25, linewidth=0.3)
    legend = axis.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=2, frameon=False, columnspacing=0.8, handletextpad=0.4)
    for text in legend.get_texts():
        if text.get_text() == "GTSparse":
            text.set_fontweight("bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    platform_label = short_gpu_name(gpu) if gpu else "the measured GPU"
    caption_text(
        fig,
        "Figure 7(b).",
        f"Latency vs. sweep count (VoxelNeXt/nuScenes, FP16, {platform_label}; Minkowski in FP32).",
        x=0.04,
        y=-0.25,
        width=64,
        fontsize=7.5,
        location="lower left",
    )
    save(fig, figures / "sensitivity.pdf")


def plot_per_frame(rows, figures: Path) -> None:
    target = "voxelnext_nuscenes_sweeps1"
    rows = [row for row in rows if row["workload"] == target and row["metric"] == "conv_only"]
    if not rows:
        return
    gpu = single_gpu(rows)
    rows = [row for row in rows if row.get("gpu") == gpu]
    fig, axis = plt.subplots(figsize=(5.5, 2.2))
    for backend in ("spconv", "torchsparse", "gtsparse"):
        selected = sorted((row for row in rows if row["gpu"] == gpu and row["backend"] == backend), key=lambda row: int(row["frame_index"]))
        if selected:
            latency = np.array([float(row["latency_ms"]) for row in selected])
            window = 20 if len(latency) >= 20 else 1
            smoothed = np.convolve(latency, np.ones(window) / window, mode="valid")
            x = np.arange(window - 1, len(latency))
            marker_every = max(1, len(smoothed) // 5)
            axis.plot(
                x,
                smoothed,
                label=LABELS[backend],
                color=LINE_COLORS[backend],
                linewidth=1.5,
                alpha=0.95,
                marker=MARKERS[backend],
                markevery=marker_every,
                markersize=6,
                markeredgewidth=1.2,
                markeredgecolor="black",
                markerfacecolor=LINE_COLORS[backend],
            )
    axis.set_xlabel("Frame index", fontsize=12)
    axis.set_ylabel("Latency (ms)", fontsize=12)
    axis.tick_params(labelsize=11)
    legend = axis.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.24), ncol=3, frameon=False, columnspacing=1.0, handletextpad=0.5)
    for text in legend.get_texts():
        if text.get_text() == "GTSparse":
            text.set_fontweight("bold")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(alpha=0.2, linewidth=0.4)
    axis.set_xlim(0, max(1, max(int(row["frame_index"]) for row in rows) + 1))
    fig.tight_layout()
    platform_label = short_gpu_name(gpu) if gpu else "the measured GPU"
    caption_text(
        fig,
        "Figure 6.",
        f"Per-frame latency over a continuous sequence (VoxelNeXt/nuScenes, sw=1, FP16, {platform_label}). All systems exhibit correlated frame-to-frame variation driven by input density, but GTSparse remains consistently below baselines throughout the sequence.",
        x=0.04,
        y=-0.25,
        width=92,
        fontsize=7.5,
        location="lower left",
    )
    save(fig, figures / "per_frame_latency.pdf")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    latency = read_csv(args.results / "latency.csv")
    template_distribution = read_csv(args.results / "template_distribution.csv")
    effective_throughput = read_csv(args.results / "effective_throughput.csv")
    time_breakdown = read_csv(args.results / "time_breakdown.csv")
    peak_memory = read_csv(args.results / "peak_memory.csv")
    per_frame_latency = read_csv(args.results / "per_frame_latency.csv")
    plot_end_to_end(latency, args.figures, "fp16")
    plot_end_to_end(latency, args.figures, "fp32")
    platform_rows = (
        template_distribution,
        effective_throughput,
        time_breakdown,
        peak_memory,
        latency,
        per_frame_latency,
    )
    gpus = sorted({row["gpu"] for rows in platform_rows for row in rows if row.get("gpu")}, key=gpu_sort_key)
    for gpu in gpus:
        figures = args.figures / gpu_path_name(gpu)
        select = lambda rows: [row for row in rows if row.get("gpu") == gpu]
        plot_template_distribution(select(template_distribution), figures)
        plot_effective_throughput(select(effective_throughput), figures)
        plot_time_breakdown(select(time_breakdown), figures)
        plot_peak_memory(select(peak_memory), figures)
        plot_ablation(select(latency), figures)
        plot_sensitivity(select(latency), figures)
        plot_per_frame(select(per_frame_latency), figures)


if __name__ == "__main__":
    main()
