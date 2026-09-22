"""Autotune for expanded-worklist 3D sparse convolution.

Profiles all valid config_ids for a given (C_in, C_out, K_vol, execution_path,
sparsity_bucket) combination. Caches results to ~/.cache/gtsparse_ew_autotune.json.
"""

import json
import time
from bisect import bisect_left
from pathlib import Path
from typing import Optional

import torch

from gtsparse import _C

CACHE_PATH = Path.home() / ".cache" / "gtsparse_ew_autotune.json"
SCHEDULED_ROW_SELECTIVE_PATH = "scheduled_row_selective"
SCHEDULED_ROW_SELECTIVE_FP16_PATH = "scheduled_row_selective_fp16"

SPARSITY_BUCKETS = [0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

WARMUP_RUNS = 5
TIMED_RUNS = 10

_cache: Optional[dict] = None
_autotune_enabled: bool = True
_CACHE_VERSION = 5


def _base_path(path: str) -> str:
    return path.split(":", 1)[0]


def _scheduled_config_count(path: str) -> int:
    base = _base_path(path)
    if base == SCHEDULED_ROW_SELECTIVE_PATH:
        return len(_C.gtsparse3d_expanded_worklist_conv3d_scheduled_config_table(2))
    if base == SCHEDULED_ROW_SELECTIVE_FP16_PATH:
        return len(_C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_fp16_config_table())
    raise ValueError(f"Unsupported scheduled path {path}")


def _config_count(path: str) -> int:
    base = _base_path(path)
    if base in {SCHEDULED_ROW_SELECTIVE_PATH, SCHEDULED_ROW_SELECTIVE_FP16_PATH}:
        return _scheduled_config_count(path)
    fixed = {
        "simt": 24,
        "tf32": 16,
        "fp16": 16,
    }
    return fixed[base]


def _is_valid_cached_config(path: str, config_id: int) -> bool:
    base = _base_path(path)
    if config_id < 0:
        return False
    if base == SCHEDULED_ROW_SELECTIVE_FP16_PATH:
        table = _C.gtsparse3d_expanded_worklist_conv3d_scheduled_bundle_fp16_config_table()
        if config_id >= len(table):
            return False
        return bool(table[config_id].get("supported", True))
    return config_id < _config_count(path)


def set_ew_autotune(enabled: bool):
    global _autotune_enabled
    _autotune_enabled = enabled


def _get_execution_path(
    dtype: torch.dtype,
    allow_tf32: bool,
    fp16_variant: str = "v1",
    simt_variant: str = "baseline",
    scheduled_reuse_mode: str = "row_selective",
    scheduled_variant: str = "position_bundle_union_prune_holes",
) -> str:
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    sm = capability[0] * 10 + capability[1]
    if simt_variant == "scheduled":
        if scheduled_reuse_mode != "row_selective":
            raise ValueError("scheduled float path only supports scheduled_reuse_mode='row_selective'")
        if dtype == torch.float16:
            if sm < 70:
                raise RuntimeError(
                    f"FP16 scheduled bundle Tensor Core path requires compute capability >= 7.0, got sm_{sm}."
                )
            return f"{SCHEDULED_ROW_SELECTIVE_FP16_PATH}:{scheduled_variant}"
        if dtype == torch.float32:
            return f"{SCHEDULED_ROW_SELECTIVE_PATH}:{scheduled_variant}"
        raise ValueError(f"Unsupported scheduled dtype {dtype}")
    if dtype == torch.float16:
        if sm < 70:
            raise RuntimeError(
                f"FP16 expanded-worklist Tensor Core path requires compute capability >= 7.0, got sm_{sm}."
            )
        if fp16_variant != "v1":
            raise ValueError(f"Unknown fp16_variant {fp16_variant!r}, expected 'v1'")
        return "fp16"
    if allow_tf32 and sm >= 80:
        return "tf32"
    return "simt"


def _bucket_ratio(active_ratio: float) -> int:
    i = bisect_left(SPARSITY_BUCKETS, active_ratio)
    if i == 0:
        return 0
    if i == len(SPARSITY_BUCKETS):
        return len(SPARSITY_BUCKETS) - 1
    if active_ratio - SPARSITY_BUCKETS[i - 1] <= SPARSITY_BUCKETS[i] - active_ratio:
        return i - 1
    return i
def _make_key(C_in: int, C_out: int, K_vol: int, path: str,
              device_id: int, bucket: int) -> str:
    return f"Cin{C_in}_Cout{C_out}_K{K_vol}_{path}_gpu{device_id}_sp{bucket}"


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


def _save_cache():
    if _cache is None:
        return
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump({"version": _CACHE_VERSION, "entries": _cache}, f, indent=2)


def _profile_config(config_id, features, weight, pairs,
                    offset_counts, N_stride, N_out) -> float:
    for _ in range(WARMUP_RUNS):
        try:
            _C.gtsparse3d_expanded_worklist_conv3d_forward(
                features, weight, None, pairs, offset_counts, N_stride,
                N_out, config_id)
        except RuntimeError:
            return float("inf")
    torch.cuda.synchronize()

    times = []
    for _ in range(TIMED_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _C.gtsparse3d_expanded_worklist_conv3d_forward(
            features, weight, None, pairs, offset_counts, N_stride,
            N_out, config_id)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]
def _profile_scheduled_config(
    config_id: int,
    reuse_mode: str,
    schedule_variant: str,
    features: torch.Tensor,
    weight: torch.Tensor,
    pairs: torch.Tensor,
    offset_counts: torch.Tensor,
    N_out: int,
    active_ratio: float,
) -> float:
    """Profile scheduled bundle kernel-only time for one config.

    Scheduled bundle layout depends on BM/grid, so each candidate still needs
    one config-specific runtime build. That build is done once up front and is
    explicitly excluded from the timed region.
    """
    if reuse_mode != "row_selective":
        raise ValueError(f"Unsupported scheduled reuse_mode {reuse_mode!r}")
    from gtsparse.sparse3d.expanded_worklist import (
        build_scheduled_bundle_runtime,
        scheduled_bundle_conv3d_forward_cuda,
    )

    try:
        (
            _resolved_config_id,
            scheduled_pairs,
            tile_offsets,
            tile_slot_states,
            total_tiles,
            _selected_bundles,
            launch_info,
            _schedule_stats,
        ) = build_scheduled_bundle_runtime(
            pairs,
            offset_counts,
            N_out,
            features=features,
            weight=weight,
            config_id=config_id,
            active_ratio=active_ratio,
            schedule_variant=schedule_variant,
            collect_schedule_stats=False,
        )
    except RuntimeError:
        return float("inf")

    def run_kernel() -> None:
        scheduled_bundle_conv3d_forward_cuda(
            features,
            weight,
            None,
            scheduled_pairs,
            tile_offsets,
            tile_slot_states,
            total_tiles,
            N_out,
            config_id,
            launch_info["grid_dim_x"],
        )

    for _ in range(WARMUP_RUNS):
        try:
            run_kernel()
        except RuntimeError:
            return float("inf")
    torch.cuda.synchronize()

    times = []
    for _ in range(TIMED_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_kernel()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def _config_offset_for_path(path: str) -> int:
    return 0


def profile_all_configs(C_in, C_out, K_vol, path, device_id,
                        features, weight, pairs, offset_counts,
                        N_stride, N_out, active_ratio) -> int:
    bucket = _bucket_ratio(active_ratio)
    key = _make_key(C_in, C_out, K_vol, path, device_id, bucket)
    ncfg = _config_count(path)

    best_id, best_time = 0, float("inf")
    # print(f"[ew_autotune] Profiling {ncfg} {path} configs for {key} ...")

    config_offset = _config_offset_for_path(path)
    for cid in range(ncfg):
        t = _profile_config(cid + config_offset, features, weight, pairs,
                            offset_counts, N_stride, N_out)
        marker = " *" if t < best_time else ""
        if t < best_time:
            best_time, best_id = t, cid
        # print(f"  config {cid:2d}: {t:.3f} ms{marker}")

    cache = _load_cache()
    cache[key] = {
        "config_id": best_id,
        "time_ms": round(best_time, 4),
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_cache()
    # print(f"[ew_autotune] Best: config {best_id} ({best_time:.3f} ms)")
    return best_id


def profile_all_scheduled_configs(
    C_in,
    C_out,
    K_vol,
    path,
    reuse_mode,
    schedule_variant,
    device_id,
    features,
    weight,
    pairs,
    offset_counts,
    N_out,
    active_ratio,
) -> int:
    bucket = _bucket_ratio(active_ratio)
    key = _make_key(C_in, C_out, K_vol, path, device_id, bucket)
    ncfg = _config_count(path)

    best_id, best_time = 0, float("inf")
    for cid in range(ncfg):
        t = _profile_scheduled_config(
            cid, reuse_mode, schedule_variant, features, weight, pairs, offset_counts, N_out, active_ratio)
        if t < best_time:
            best_time, best_id = t, cid

    cache = _load_cache()
    cache[key] = {
        "config_id": best_id,
        "time_ms": round(best_time, 4),
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_cache()
    return best_id


def get_config_id(C_in, C_out, K_vol, dtype, features, weight,
                  pairs, offset_counts, N_stride, N_out,
                  active_ratio, fp16_variant: str = "v1",
                  simt_variant: str = "baseline",
                  scheduled_reuse_mode: str = "row_selective",
                  scheduled_variant: str = "position_bundle_union_prune_holes") -> int:
    if not _autotune_enabled:
        return 0

    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    path = _get_execution_path(
        dtype,
        allow_tf32,
        fp16_variant=fp16_variant,
        simt_variant=simt_variant,
        scheduled_reuse_mode=scheduled_reuse_mode,
        scheduled_variant=scheduled_variant,
    )
    device_id = features.device.index or 0
    bucket = _bucket_ratio(active_ratio)
    key = _make_key(C_in, C_out, K_vol, path, device_id, bucket)

    cache = _load_cache()
    if key in cache:
        cached_id = int(cache[key]["config_id"])
        if _is_valid_cached_config(path, cached_id):
            return cached_id + _config_offset_for_path(path)
        del cache[key]
        _save_cache()

    if path.startswith(SCHEDULED_ROW_SELECTIVE_PATH + ":") or path.startswith(SCHEDULED_ROW_SELECTIVE_FP16_PATH + ":"):
        best_id = profile_all_scheduled_configs(
            C_in, C_out, K_vol, path, scheduled_reuse_mode, scheduled_variant, device_id,
            features, weight, pairs, offset_counts, N_out, active_ratio,
        )
        return best_id

    best_id = profile_all_configs(
        C_in, C_out, K_vol, path, device_id,
        features, weight, pairs, offset_counts, N_stride,
        N_out, active_ratio,
    )
    return best_id + _config_offset_for_path(path)


def get_cached_config_id(
    C_in,
    C_out,
    K_vol,
    dtype,
    features,
    *,
    active_ratio,
    fp16_variant: str = "v1",
    simt_variant: str = "baseline",
    scheduled_reuse_mode: str = "row_selective",
    scheduled_variant: str = "position_bundle_union_prune_holes",
) -> int | None:
    """Return a cached config id if available, without profiling."""
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    path = _get_execution_path(
        dtype,
        allow_tf32,
        fp16_variant=fp16_variant,
        simt_variant=simt_variant,
        scheduled_reuse_mode=scheduled_reuse_mode,
        scheduled_variant=scheduled_variant,
    )
    device_id = features.device.index or 0
    bucket = _bucket_ratio(active_ratio)
    key = _make_key(C_in, C_out, K_vol, path, device_id, bucket)
    cache = _load_cache()
    entry = cache.get(key)
    if entry is None:
        return None
    cached_id = int(entry["config_id"])
    if not _is_valid_cached_config(path, cached_id):
        del cache[key]
        _save_cache()
        return None
    return cached_id + _config_offset_for_path(path)
