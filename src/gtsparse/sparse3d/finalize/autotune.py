"""Autotune for finalized FP16 flattened-worklist variants."""

from __future__ import annotations

import json
import time
from bisect import bisect_left
from pathlib import Path
from typing import Sequence

import torch


CACHE_PATH = Path.home() / ".cache" / "gtsparse_finalize_autotune.json"
SPARSITY_BUCKETS = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
WARMUP_RUNS = 5
TIMED_RUNS = 10
_CACHE_VERSION = 4
_cache: dict | None = None
_autotune_enabled = True

FINALIZE_FP16_VARIANT_ORDER = ("naive", "sticky", "bundle")
FINALIZE_FP16_BUNDLE_SIZES = (2, 4, 8)


def set_finalize_autotune(enabled: bool) -> None:
    global _autotune_enabled
    _autotune_enabled = bool(enabled)


def _bucket_ratio(active_ratio: float) -> int:
    i = bisect_left(SPARSITY_BUCKETS, active_ratio)
    if i == 0:
        return 0
    if i == len(SPARSITY_BUCKETS):
        return len(SPARSITY_BUCKETS) - 1
    if active_ratio - SPARSITY_BUCKETS[i - 1] <= SPARSITY_BUCKETS[i] - active_ratio:
        return i - 1
    return i


def _load_cache() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH) as f:
                data = json.load(f)
            _cache = data.get("entries", {}) if data.get("version") == _CACHE_VERSION else {}
        except (json.JSONDecodeError, KeyError):
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"version": _CACHE_VERSION, "entries": _cache}, f, indent=2)


def _normalize_candidate_variants(runtime_bundle_size: int, candidate_variants: Sequence[str] | None, has_bundle_runtime_map: bool) -> tuple[str, ...]:
    if candidate_variants is None:
        if has_bundle_runtime_map:
            return ("naive", "sticky", "bundle")
        if runtime_bundle_size <= 1:
            return ("naive", "sticky")
        return ("bundle",)
    normalized = tuple(str(v) for v in candidate_variants)
    for variant in normalized:
        if variant not in FINALIZE_FP16_VARIANT_ORDER:
            raise ValueError(f"unknown finalized fp16 variant {variant!r}")
    return normalized


def _normalize_candidate_bundle_sizes(candidate_bundle_sizes: Sequence[int] | None) -> tuple[int, ...]:
    if candidate_bundle_sizes is None:
        return FINALIZE_FP16_BUNDLE_SIZES
    normalized = tuple(int(v) for v in candidate_bundle_sizes)
    for bundle_size in normalized:
        if bundle_size not in FINALIZE_FP16_BUNDLE_SIZES:
            raise ValueError(f"unsupported finalized bundle_size {bundle_size}; expected one of {FINALIZE_FP16_BUNDLE_SIZES}")
    return normalized


def _finalized_fp16_variant_space() -> list[dict[str, int | str | None]]:
    from .worklist import (
        finalized_naive_config_table,
        finalized_static_bundle_config_table,
        finalized_sticky_config_table,
    )

    cursor = 0
    entries: list[dict[str, int | str | None]] = []
    scalar_counts = {
        "naive": len(finalized_naive_config_table(dtype=torch.float16)),
        "sticky": len(finalized_sticky_config_table(dtype=torch.float16)),
    }
    for variant in ("naive", "sticky"):
        count = scalar_counts[variant]
        entries.append({"variant": variant, "bundle_size": None, "offset": cursor, "count": count})
        cursor += count
    bundle_count = len(finalized_static_bundle_config_table())
    for bundle_size in FINALIZE_FP16_BUNDLE_SIZES:
        entries.append({"variant": "bundle", "bundle_size": int(bundle_size), "offset": cursor, "count": bundle_count})
        cursor += bundle_count
    return entries


def encode_finalize_fp16_config_id(variant: str, local_config_id: int, *, bundle_size: int | None = None) -> int:
    space = _finalized_fp16_variant_space()
    local = int(local_config_id)
    for entry in space:
        if str(entry["variant"]) == variant and entry["bundle_size"] == bundle_size:
            count = int(entry["count"])
            if local < 0 or local >= count:
                raise ValueError(f"local_config_id {local} is out of range for variant {variant} bundle_size={bundle_size}")
            return int(entry["offset"]) + local
    raise ValueError(f"unknown finalized fp16 variant {variant!r} bundle_size={bundle_size}")


def decode_finalize_fp16_config_id(config_id: int) -> tuple[str, int | None, int]:
    cid = int(config_id)
    space = _finalized_fp16_variant_space()
    for entry in space:
        start = int(entry["offset"])
        count = int(entry["count"])
        if start <= cid < start + count:
            return str(entry["variant"]), entry["bundle_size"] if entry["bundle_size"] is None else int(entry["bundle_size"]), cid - start
    raise ValueError(f"finalized fp16 config_id {cid} is out of shared-space range")


def _key(
    C_in: int,
    C_out: int,
    K_vol: int,
    bundle_size: int,
    variants: tuple[str, ...],
    bundle_sizes: tuple[int, ...],
    device_id: int,
    bucket: int,
) -> str:
    return (
        f"Cin{C_in}_Cout{C_out}_K{K_vol}_bs{bundle_size}_variants{','.join(variants)}_bundle_sizes{','.join(str(v) for v in bundle_sizes)}"
        f"_gpu{device_id}_sp{bucket}"
    )


def _profile_variant(
    variant: str,
    bundle_size: int | None,
    local_config_id: int,
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime,
    bundle_runtime_map: dict[int, object] | None,
) -> float:
    from .worklist import (
        finalized_naive_worklist_conv3d_cuda,
        finalized_static_bundle_conv3d_cuda,
        finalized_sticky_offset_conv3d_cuda,
    )

    if variant == "naive":
        run = lambda: finalized_naive_worklist_conv3d_cuda(features, weight, None, runtime, config_id=local_config_id)
    elif variant == "sticky":
        run = lambda: finalized_sticky_offset_conv3d_cuda(features, weight, None, runtime, config_id=local_config_id)
    elif variant == "bundle":
        if bundle_size is None or bundle_runtime_map is None or bundle_size not in bundle_runtime_map:
            return float("inf")
        bundle_runtime = bundle_runtime_map[bundle_size]
        run = lambda: finalized_static_bundle_conv3d_cuda(features, weight, None, bundle_runtime, config_id=local_config_id)
    else:
        raise ValueError(f"unknown finalized fp16 variant {variant!r}")

    for _ in range(WARMUP_RUNS):
        try:
            run()
        except RuntimeError:
            return float("inf")
    torch.cuda.synchronize()

    times = []
    for _ in range(TIMED_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    times.sort()
    return times[len(times) // 2]


def get_finalize_fp16_config_id(
    C_in: int,
    C_out: int,
    K_vol: int,
    features: torch.Tensor,
    weight: torch.Tensor,
    runtime,
    *,
    active_ratio: float,
    candidate_variants: Sequence[str] | None = None,
    candidate_bundle_sizes: Sequence[int] | None = None,
    bundle_runtime_map: dict[int, object] | None = None,
) -> int:
    variants = _normalize_candidate_variants(int(runtime.bundle_size), candidate_variants, bundle_runtime_map is not None and len(bundle_runtime_map) > 0)
    bundle_sizes = _normalize_candidate_bundle_sizes(candidate_bundle_sizes)
    if not _autotune_enabled:
        if variants[0] == "bundle":
            return encode_finalize_fp16_config_id("bundle", 0, bundle_size=bundle_sizes[0])
        return encode_finalize_fp16_config_id(variants[0], 0)

    device_id = features.device.index or 0
    bucket = _bucket_ratio(active_ratio)
    cache_key = _key(C_in, C_out, K_vol, int(runtime.bundle_size), variants, bundle_sizes, device_id, bucket)
    cache = _load_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        cid = int(cached["config_id"])
        variant, bundle_size, _local = decode_finalize_fp16_config_id(cid)
        if variant in variants and (variant != "bundle" or bundle_size in bundle_sizes):
            return cid
        del cache[cache_key]
        _save_cache()

    space = _finalized_fp16_variant_space()
    if variants[0] == "bundle":
        best_config = encode_finalize_fp16_config_id("bundle", 0, bundle_size=bundle_sizes[0])
    else:
        best_config = encode_finalize_fp16_config_id(variants[0], 0)
    best_time = float("inf")
    for entry in space:
        variant = str(entry["variant"])
        entry_bundle_size = entry["bundle_size"] if entry["bundle_size"] is None else int(entry["bundle_size"])
        if variant not in variants:
            continue
        if variant == "bundle" and entry_bundle_size not in bundle_sizes:
            continue
        count = int(entry["count"])
        for local in range(count):
            t = _profile_variant(variant, entry_bundle_size, local, features, weight, runtime, bundle_runtime_map)
            if t < best_time:
                best_time = t
                if variant == "bundle":
                    best_config = encode_finalize_fp16_config_id(variant, local, bundle_size=entry_bundle_size)
                else:
                    best_config = encode_finalize_fp16_config_id(variant, local)

    cache[cache_key] = {
        "config_id": int(best_config),
        "time_ms": round(best_time, 4),
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_cache()
    return int(best_config)
