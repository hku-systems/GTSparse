"""KITTI + SECOND thin end-to-end surface for GTSparse and baselines."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torchvision.ops
from tqdm.auto import tqdm

import torchsparse

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torchsparse.backends.allow_tf32 = False

from .common import clear_sparse_metadata, measure_cuda_elapsed_ms, require_cuda_device, resolve_runtime_dtype
from gtsparse.sparse3d.geometric_template import (
    GeometricTemplateKernel3Conv3d,
    GeometricTemplateSparseConv3d,
    GeometricTemplateSubMConv3d,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor



_CPU_POINT_TO_VOXEL_CACHE: dict[tuple[tuple[float, float, float], tuple[float, float, float, float, float, float], int, int, int], object] = {}


def _make_torchsparse_conv_config(*, sorted_flag: bool):
    from torchsparse.nn.functional.conv.conv_config import get_default_conv_config

    base = get_default_conv_config()
    config = type(base)(dict(base))
    config.ifsort = bool(sorted_flag)
    return config


def _configure_torchsparse_conv_sort_mode(sorted_flag: bool) -> None:
    from torchsparse.nn.functional.conv.conv_config import set_global_conv_config

    set_global_conv_config(_make_torchsparse_conv_config(sorted_flag=bool(sorted_flag)))


@dataclass(frozen=True)
class KittiSecondDataConfig:
    root: Path = Path("dataset/kitti")
    split: str = "test"
    class_names: tuple[str, ...] = ("Car", "Pedestrian", "Cyclist")
    point_cloud_range: tuple[float, float, float, float, float, float] = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0)
    voxel_size: tuple[float, float, float] = (0.05, 0.05, 0.1)
    num_point_features: int = 4
    max_points_per_voxel: int = 5
    max_voxels_train: int = 16000
    max_voxels_eval: int = 40000
    max_sweeps: int = 1

    @property
    def grid_size_xyz(self) -> tuple[int, int, int]:
        ranges = np.asarray(self.point_cloud_range[3:], dtype=np.float32) - np.asarray(self.point_cloud_range[:3], dtype=np.float32)
        voxel = np.asarray(self.voxel_size, dtype=np.float32)
        grid = np.round(ranges / voxel).astype(np.int64)
        return int(grid[0]), int(grid[1]), int(grid[2])

    @property
    def sparse_shape_zyx(self) -> tuple[int, int, int]:
        gx, gy, gz = self.grid_size_xyz
        return int(gz + 1), int(gy), int(gx)

    def max_voxels(self, *, training: bool) -> int:
        return int(self.max_voxels_train if training else self.max_voxels_eval)


@dataclass(frozen=True)
class KittiSecondModelConfig:
    input_channels: int = 4
    conv_input_channels: int = 16
    conv2_channels: int = 32
    conv3_channels: int = 64
    conv4_channels: int = 64
    conv_out_channels: int = 128
    bev_layer_nums: tuple[int, int] = (5, 5)
    bev_layer_strides: tuple[int, int] = (1, 2)
    bev_num_filters: tuple[int, int] = (128, 256)
    bev_upsample_strides: tuple[int, int] = (1, 2)
    bev_num_upsample_filters: tuple[int, int] = (256, 256)
    anchor_rotations: tuple[float, ...] = (0.0, 1.57)
    anchor_sizes: tuple[tuple[float, float, float], ...] = (
        (3.9, 1.6, 1.56),
        (0.8, 0.6, 1.73),
        (1.76, 0.6, 1.73),
    )
    anchor_bottom_heights: tuple[float, ...] = (-1.78, -0.6, -0.6)
    anchor_feature_map_stride: int = 8
    box_code_size: int = 7
    num_dir_bins: int = 2

    @property
    def num_bev_input_channels(self) -> int:
        return int(self.conv_out_channels * 2)

    def num_anchors_per_location(self, class_names: Sequence[str]) -> int:
        return int(len(class_names) * len(self.anchor_rotations))


@dataclass(frozen=True)
class KittiSecondConfig:
    data: KittiSecondDataConfig = field(default_factory=KittiSecondDataConfig)
    model: KittiSecondModelConfig = field(default_factory=KittiSecondModelConfig)


def discover_kitti_root(root: str | Path | None = None) -> Path:
    if root is None:
        return Path("dataset/kitti")
    return Path(root)


def read_kitti_points(path: str | Path) -> np.ndarray:
    points = np.fromfile(str(path), dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"KITTI lidar file {path} does not contain a multiple of 4 float32 values")
    return points.reshape(-1, 4)


@dataclass(frozen=True)
class KittiLidarSample:
    frame_id: str
    points: np.ndarray
    image_shape: tuple[int, int] | None = None
    calib_lines: tuple[str, ...] | None = None
    label_lines: tuple[str, ...] | None = None


class KittiLidarDataset(torch_data.Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "val",
        include_labels: bool = False,
        include_calib: bool = False,
    ) -> None:
        super().__init__()
        self.root = discover_kitti_root(root)
        self.split = str(split)
        self.include_labels = bool(include_labels)
        self.include_calib = bool(include_calib)
        self.root_split = self.root / ("training" if self.split != "test" else "testing")
        self.sample_ids = self._load_sample_ids()

    def _load_sample_ids(self) -> list[str]:
        split_file = self.root / "ImageSets" / f"{self.split}.txt"
        if split_file.exists():
            return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        velodyne_dir = self.root_split / "velodyne"
        if velodyne_dir.exists():
            return sorted(path.stem for path in velodyne_dir.glob("*.bin"))
        raise FileNotFoundError(
            f"Could not find KITTI split file {split_file} or extracted velodyne directory {velodyne_dir}. "
            f"Please unpack the velodyne point clouds under {self.root_split}."
        )

    def _resolve_lidar_path(self, frame_id: str) -> Path | None:
        lidar_path = self.root_split / "velodyne" / f"{frame_id}.bin"
        return lidar_path if lidar_path.exists() else None

    def _load_lidar_points(self, frame_id: str) -> np.ndarray:
        lidar_path = self._resolve_lidar_path(frame_id)
        if lidar_path is not None:
            return read_kitti_points(lidar_path)
        raise FileNotFoundError(
            f"Could not find lidar for frame {frame_id} under {self.root_split}. "
            f"Expected extracted file {self.root_split / 'velodyne' / f'{frame_id}.bin'}."
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _read_text_file(self, path: Path) -> tuple[str, ...] | None:
        if not path.exists():
            return None
        return tuple(line.rstrip("\n") for line in path.read_text().splitlines())

    def __getitem__(self, index: int) -> dict[str, Any]:
        frame_id = self.sample_ids[index]
        points = self._load_lidar_points(frame_id)
        image_shape = None
        image_path = self.root_split / "image_2" / f"{frame_id}.png"
        if image_path.exists():
            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    image_shape = (int(image.height), int(image.width))
            except Exception:
                image_shape = None
        sample = KittiLidarSample(
            frame_id=frame_id,
            points=points,
            image_shape=image_shape,
            calib_lines=self._read_text_file(self.root_split / "calib" / f"{frame_id}.txt") if self.include_calib else None,
            label_lines=self._read_text_file(self.root_split / "label_2" / f"{frame_id}.txt") if self.include_labels else None,
        )
        return {
            "frame_id": sample.frame_id,
            "points": sample.points,
            "image_shape": sample.image_shape,
            "calib_lines": sample.calib_lines,
            "label_lines": sample.label_lines,
        }


@dataclass
class KittiSecondBatch:
    frame_ids: list[str]
    points: list[np.ndarray]
    voxels: torch.Tensor
    voxel_coords: torch.Tensor
    voxel_num_points: torch.Tensor
    batch_size: int
    grid_size_xyz: tuple[int, int, int]
    sparse_shape_zyx: tuple[int, int, int]


def _get_cpu_point_to_voxel(
    *,
    voxel_size: tuple[float, float, float],
    point_cloud_range: tuple[float, float, float, float, float, float],
    num_point_features: int,
    max_voxels: int,
    max_points_per_voxel: int,
):
    key = (
        tuple(float(v) for v in voxel_size),
        tuple(float(v) for v in point_cloud_range),
        int(num_point_features),
        int(max_voxels),
        int(max_points_per_voxel),
    )
    generator = _CPU_POINT_TO_VOXEL_CACHE.get(key)
    if generator is None:
        from spconv.pytorch.utils import PointToVoxel

        generator = PointToVoxel(
            vsize_xyz=list(key[0]),
            coors_range_xyz=list(key[1]),
            num_point_features=key[2],
            max_num_voxels=key[3],
            max_num_points_per_voxel=key[4],
            device=torch.device("cpu"),
        )
        _CPU_POINT_TO_VOXEL_CACHE[key] = generator
    return generator


def _hard_voxelize_points_numpy(
    points: np.ndarray,
    *,
    voxel_size: tuple[float, float, float],
    point_cloud_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be [N, >=3]")

    voxel = np.asarray(voxel_size, dtype=np.float32)
    pc_min = np.asarray(point_cloud_range[:3], dtype=np.float32)
    pc_max = np.asarray(point_cloud_range[3:], dtype=np.float32)
    grid_size_xyz = np.floor((pc_max - pc_min) / voxel).astype(np.int64)
    xyz = points[:, :3]
    in_range = np.all((xyz >= pc_min) & (xyz < pc_max), axis=1)
    filtered = points[in_range]
    if filtered.shape[0] == 0:
        empty_voxels = np.empty((0, max_points_per_voxel, points.shape[1]), dtype=np.float32)
        empty_coords = np.empty((0, 3), dtype=np.int32)
        empty_num_points = np.empty((0,), dtype=np.int32)
        return empty_voxels, empty_coords, empty_num_points

    voxel_indices_xyz = np.floor((filtered[:, :3] - pc_min) / voxel).astype(np.int32)
    key_xyz = voxel_indices_xyz.astype(np.int64)
    flat_keys = key_xyz[:, 0] + grid_size_xyz[0] * (key_xyz[:, 1] + grid_size_xyz[1] * key_xyz[:, 2])
    unique_keys, first_idx, inverse = np.unique(flat_keys, return_index=True, return_inverse=True)
    if unique_keys.size == 0:
        empty_voxels = np.empty((0, max_points_per_voxel, points.shape[1]), dtype=np.float32)
        empty_coords = np.empty((0, 3), dtype=np.int32)
        empty_num_points = np.empty((0,), dtype=np.int32)
        return empty_voxels, empty_coords, empty_num_points

    voxel_order = np.argsort(first_idx, kind="stable")
    selected_unique = voxel_order[: min(int(unique_keys.size), int(max_voxels))]
    remap = np.full((int(unique_keys.size),), -1, dtype=np.int32)
    remap[selected_unique] = np.arange(int(selected_unique.size), dtype=np.int32)
    voxel_ids_all = remap[inverse]
    keep_points = voxel_ids_all >= 0
    kept_points = filtered[keep_points]
    kept_indices = voxel_indices_xyz[keep_points]
    kept_voxel_ids = voxel_ids_all[keep_points]
    point_order = np.argsort(kept_voxel_ids, kind="stable")
    sorted_points = kept_points[point_order]
    sorted_indices = kept_indices[point_order]
    voxel_ids = kept_voxel_ids[point_order]
    counts = np.bincount(voxel_ids, minlength=int(selected_unique.size))
    first_offsets = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.int32))) if counts.size > 0 else np.empty((0,), dtype=np.int32)
    total_sorted_points = int(counts.sum())
    repeated_starts = np.repeat(first_offsets, counts)
    ranks = np.arange(total_sorted_points, dtype=np.int32) - repeated_starts
    keep = ranks < int(max_points_per_voxel)

    voxel_count = int(selected_unique.size)
    voxels = np.zeros((voxel_count, max_points_per_voxel, filtered.shape[1]), dtype=np.float32)
    voxels[voxel_ids[keep], ranks[keep]] = sorted_points[:total_sorted_points][keep]
    coords = voxel_indices_xyz[first_idx[selected_unique]][:, [2, 1, 0]].astype(np.int32, copy=False)
    num_points = np.minimum(counts, int(max_points_per_voxel)).astype(np.int32, copy=False)
    return voxels, coords, num_points


def hard_voxelize_points(
    points: np.ndarray,
    *,
    voxel_size: tuple[float, float, float],
    point_cloud_range: tuple[float, float, float, float, float, float],
    max_points_per_voxel: int,
    max_voxels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must be [N, >=3]")
    try:
        generator = _get_cpu_point_to_voxel(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            num_point_features=int(points.shape[1]),
            max_voxels=int(max_voxels),
            max_points_per_voxel=int(max_points_per_voxel),
        )
        voxels, coords, num_points = generator(torch.from_numpy(np.asarray(points, dtype=np.float32)))
        return voxels.numpy(), coords.numpy(), num_points.numpy()
    except Exception:
        return _hard_voxelize_points_numpy(
            points,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
        )


def collate_kitti_second_batch(
    samples: list[dict[str, Any]],
    data_cfg: KittiSecondDataConfig,
    *,
    training: bool = False,
) -> KittiSecondBatch:
    frame_ids: list[str] = []
    points_list: list[np.ndarray] = []
    voxel_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    total_voxels = 0

    for batch_idx, sample in enumerate(samples):
        points = np.asarray(sample["points"], dtype=np.float32)
        voxels, coords_zyx, num_points = hard_voxelize_points(
            points,
            voxel_size=data_cfg.voxel_size,
            point_cloud_range=data_cfg.point_cloud_range,
            max_points_per_voxel=data_cfg.max_points_per_voxel,
            max_voxels=data_cfg.max_voxels(training=training),
        )
        voxel_chunks.append((voxels, coords_zyx, num_points))
        total_voxels += int(voxels.shape[0])
        frame_ids.append(str(sample["frame_id"]))
        points_list.append(points)

    num_features = int(samples[0]["points"].shape[1]) if samples else int(data_cfg.num_point_features)
    if voxel_chunks:
        voxels = np.empty((total_voxels, data_cfg.max_points_per_voxel, num_features), dtype=np.float32)
        voxel_coords = np.empty((total_voxels, 4), dtype=np.int32)
        voxel_num_points = np.empty((total_voxels,), dtype=np.int32)
        cursor = 0
        for batch_idx, (voxel_chunk, coords_zyx, num_points) in enumerate(voxel_chunks):
            count = int(voxel_chunk.shape[0])
            if count == 0:
                continue
            end = cursor + count
            voxels[cursor:end] = voxel_chunk
            voxel_coords[cursor:end, 0] = batch_idx
            voxel_coords[cursor:end, 1:] = coords_zyx
            voxel_num_points[cursor:end] = num_points
            cursor = end
    else:
        voxels = np.empty((0, data_cfg.max_points_per_voxel, num_features), dtype=np.float32)
        voxel_coords = np.empty((0, 4), dtype=np.int32)
        voxel_num_points = np.empty((0,), dtype=np.int32)

    return KittiSecondBatch(
        frame_ids=frame_ids,
        points=points_list,
        voxels=torch.from_numpy(voxels).float(),
        voxel_coords=torch.from_numpy(voxel_coords).int(),
        voxel_num_points=torch.from_numpy(voxel_num_points).int(),
        batch_size=len(samples),
        grid_size_xyz=data_cfg.grid_size_xyz,
        sparse_shape_zyx=data_cfg.sparse_shape_zyx,
    )


def move_kitti_second_batch_to_device(
    batch: KittiSecondBatch,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
) -> KittiSecondBatch:
    voxels = batch.voxels.to(device) if dtype is None else batch.voxels.to(device=device, dtype=dtype)
    return KittiSecondBatch(
        frame_ids=batch.frame_ids,
        points=batch.points,
        voxels=voxels,
        voxel_coords=batch.voxel_coords.to(device),
        voxel_num_points=batch.voxel_num_points.to(device),
        batch_size=batch.batch_size,
        grid_size_xyz=batch.grid_size_xyz,
        sparse_shape_zyx=batch.sparse_shape_zyx,
    )


def _limit_period(val: torch.Tensor, *, offset: float = 0.0, period: float) -> torch.Tensor:
    return val - torch.floor(val / period + offset) * period


def generate_second_anchors(
    *,
    data_cfg: KittiSecondDataConfig,
    model_cfg: KittiSecondModelConfig,
    feature_map_size_hw: tuple[int, int],
    device: torch.device | str,
) -> torch.Tensor:
    h, w = int(feature_map_size_hw[0]), int(feature_map_size_hw[1])
    pc_range = data_cfg.point_cloud_range
    x_stride = (pc_range[3] - pc_range[0]) / max(w - 1, 1)
    y_stride = (pc_range[4] - pc_range[1]) / max(h - 1, 1)

    x_shifts = torch.arange(pc_range[0], pc_range[3] + 1e-5, step=x_stride, dtype=torch.float32, device=device)
    y_shifts = torch.arange(pc_range[1], pc_range[4] + 1e-5, step=y_stride, dtype=torch.float32, device=device)
    if int(x_shifts.numel()) > w:
        x_shifts = x_shifts[:w]
    if int(y_shifts.numel()) > h:
        y_shifts = y_shifts[:h]
    while int(x_shifts.numel()) < w:
        x_shifts = torch.cat((x_shifts, x_shifts.new_tensor([pc_range[3]])))
    while int(y_shifts.numel()) < h:
        y_shifts = torch.cat((y_shifts, y_shifts.new_tensor([pc_range[4]])))

    yy, xx = torch.meshgrid(y_shifts, x_shifts, indexing="ij")
    anchor_sets = []
    for size, bottom_height in zip(model_cfg.anchor_sizes, model_cfg.anchor_bottom_heights):
        dx, dy, dz = size
        center_z = float(bottom_height) + float(dz) / 2.0
        per_rot = []
        for rot in model_cfg.anchor_rotations:
            anchor = torch.stack(
                (
                    xx,
                    yy,
                    xx.new_full(xx.shape, center_z),
                    xx.new_full(xx.shape, float(dx)),
                    xx.new_full(xx.shape, float(dy)),
                    xx.new_full(xx.shape, float(dz)),
                    xx.new_full(xx.shape, float(rot)),
                ),
                dim=-1,
            )
            per_rot.append(anchor)
        anchor_sets.append(torch.stack(per_rot, dim=2))
    return torch.cat(anchor_sets, dim=2).contiguous()


def decode_second_boxes(box_encodings: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    xa, ya, za, dxa, dya, dza, ra = torch.split(anchors, 1, dim=-1)
    xt, yt, zt, dxt, dyt, dzt, rt = torch.split(box_encodings, 1, dim=-1)
    diagonal = torch.sqrt(dxa ** 2 + dya ** 2)
    xg = xt * diagonal + xa
    yg = yt * diagonal + ya
    zg = zt * dza + za
    dxg = torch.exp(dxt) * dxa
    dyg = torch.exp(dyt) * dya
    dzg = torch.exp(dzt) * dza
    rg = rt + ra
    return torch.cat((xg, yg, zg, dxg, dyg, dzg, rg), dim=-1)


@dataclass(frozen=True)
class DecodedSecondPredictions:
    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


def decode_second_predictions(
    predictions: dict[str, torch.Tensor],
    *,
    data_cfg: KittiSecondDataConfig,
    model_cfg: KittiSecondModelConfig,
) -> DecodedSecondPredictions:
    cls_preds = predictions["cls_preds"]
    box_preds = predictions["box_preds"]
    dir_cls_preds = predictions["dir_cls_preds"]
    batch_size, h, w, _cls_channels = cls_preds.shape
    num_classes = len(data_cfg.class_names)
    num_anchors = model_cfg.num_anchors_per_location(data_cfg.class_names)

    cls_preds = cls_preds.view(batch_size, h, w, num_anchors, num_classes)
    box_preds = box_preds.view(batch_size, h, w, num_anchors, model_cfg.box_code_size)
    dir_cls_preds = dir_cls_preds.view(batch_size, h, w, num_anchors, model_cfg.num_dir_bins)

    anchors = generate_second_anchors(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        feature_map_size_hw=(h, w),
        device=cls_preds.device,
    ).view(1, h, w, num_anchors, model_cfg.box_code_size)
    decoded_boxes = decode_second_boxes(box_preds, anchors)

    dir_labels = torch.argmax(dir_cls_preds, dim=-1)
    period = (2.0 * torch.pi) / float(model_cfg.num_dir_bins)
    dir_rot = _limit_period(decoded_boxes[..., 6], offset=0.0, period=period)
    decoded_boxes[..., 6] = dir_rot + period * dir_labels.to(decoded_boxes.dtype)

    scores, labels = torch.max(torch.sigmoid(cls_preds), dim=-1)
    labels = labels + 1
    return DecodedSecondPredictions(
        boxes=decoded_boxes.view(batch_size, -1, model_cfg.box_code_size),
        scores=scores.view(batch_size, -1),
        labels=labels.view(batch_size, -1),
    )


def select_topk_predictions(decoded: DecodedSecondPredictions, *, topk: int) -> list[dict[str, torch.Tensor]]:
    outputs: list[dict[str, torch.Tensor]] = []
    for batch_idx in range(int(decoded.scores.size(0))):
        scores = decoded.scores[batch_idx]
        count = min(int(topk), int(scores.numel()))
        topk_scores, topk_idx = torch.topk(scores, k=count, sorted=True)
        outputs.append(
            {
                "pred_scores": topk_scores,
                "pred_labels": decoded.labels[batch_idx].index_select(0, topk_idx),
                "pred_boxes": decoded.boxes[batch_idx].index_select(0, topk_idx),
            }
        )
    return outputs


def _boxes3d_to_bev_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x = boxes[:, 0]
    y = boxes[:, 1]
    dx = torch.clamp_min(boxes[:, 3], 1e-4)
    dy = torch.clamp_min(boxes[:, 4], 1e-4)
    half_dx = dx * 0.5
    half_dy = dy * 0.5
    return torch.stack((x - half_dx, y - half_dy, x + half_dx, y + half_dy), dim=-1)


def select_nms_predictions(
    decoded: DecodedSecondPredictions,
    *,
    score_thresh: float = 0.1,
    nms_thresh: float = 0.01,
    pre_maxsize: int = 4096,
    post_maxsize: int = 500,
) -> list[dict[str, torch.Tensor]]:
    outputs: list[dict[str, torch.Tensor]] = []
    num_classes = int(decoded.labels.max().item()) if decoded.labels.numel() > 0 else 0
    for batch_idx in range(int(decoded.scores.size(0))):
        batch_boxes = decoded.boxes[batch_idx]
        batch_scores = decoded.scores[batch_idx]
        batch_labels = decoded.labels[batch_idx]
        kept_scores: list[torch.Tensor] = []
        kept_labels: list[torch.Tensor] = []
        kept_boxes: list[torch.Tensor] = []
        for label in range(1, num_classes + 1):
            mask = (batch_labels == label) & (batch_scores >= float(score_thresh))
            if not bool(mask.any()):
                continue
            label_scores = batch_scores[mask]
            label_boxes = batch_boxes[mask]
            if int(label_scores.numel()) > int(pre_maxsize):
                top_scores, top_idx = torch.topk(label_scores, k=int(pre_maxsize), sorted=True)
                label_scores = top_scores
                label_boxes = label_boxes.index_select(0, top_idx)
            keep = torchvision.ops.nms(_boxes3d_to_bev_xyxy(label_boxes), label_scores, float(nms_thresh))
            if int(keep.numel()) > int(post_maxsize):
                keep = keep[: int(post_maxsize)]
            kept_scores.append(label_scores.index_select(0, keep))
            kept_labels.append(torch.full((int(keep.numel()),), label, dtype=batch_labels.dtype, device=batch_labels.device))
            kept_boxes.append(label_boxes.index_select(0, keep))
        if kept_scores:
            scores = torch.cat(kept_scores, dim=0)
            labels = torch.cat(kept_labels, dim=0)
            boxes = torch.cat(kept_boxes, dim=0)
            order = torch.argsort(scores, descending=True)
            if int(order.numel()) > int(post_maxsize):
                order = order[: int(post_maxsize)]
            outputs.append(
                {
                    "pred_scores": scores.index_select(0, order),
                    "pred_labels": labels.index_select(0, order),
                    "pred_boxes": boxes.index_select(0, order),
                }
            )
        else:
            outputs.append(
                {
                    "pred_scores": batch_scores.new_empty((0,)),
                    "pred_labels": batch_labels.new_empty((0,), dtype=batch_labels.dtype),
                    "pred_boxes": batch_boxes.new_empty((0, batch_boxes.size(-1))),
                }
            )
    return outputs


def _donor_sparse_backbone_conv_keys() -> list[str]:
    stage_block_counts = {"conv1": 1, "conv2": 3, "conv3": 3, "conv4": 3}
    keys = ["backbone_3d.conv_input.0.weight"]
    for stage_name, num_blocks in stage_block_counts.items():
        for block_idx in range(num_blocks):
            keys.append(f"backbone_3d.{stage_name}.{block_idx}.0.weight")
    keys.append("backbone_3d.conv_out.0.weight")
    return keys


def _donor_sparse_backbone_bn_keys() -> list[str]:
    stage_block_counts = {"conv1": 1, "conv2": 3, "conv3": 3, "conv4": 3}
    keys = []
    for prefix in ["backbone_3d.conv_input.1"] + [f"backbone_3d.{stage}.{idx}.1" for stage, count in stage_block_counts.items() for idx in range(count)] + ["backbone_3d.conv_out.1"]:
        keys.extend([
            prefix + ".weight",
            prefix + ".bias",
            prefix + ".running_mean",
            prefix + ".running_var",
            prefix + ".num_batches_tracked",
        ])
    return keys


def _target_sparse_backbone_conv_keys(model: torch.nn.Module) -> list[str]:
    state = model.state_dict()
    return [key for key in state.keys() if key.startswith("sparse_backbone.") and state[key].ndim >= 3]


def _target_sparse_backbone_bn_keys(model: torch.nn.Module) -> list[str]:
    state = model.state_dict()
    return [key for key in state.keys() if key.startswith("sparse_backbone.") and state[key].ndim < 3]


def _reorder_k_offsets_if_needed(weights: torch.Tensor) -> torch.Tensor:
    if int(weights.size(0)) != 27:
        return weights
    ky_kx = 9
    order = [x * ky_kx + y * 3 + z for z in range(3) for y in range(3) for x in range(3)]
    index = torch.tensor(order, dtype=torch.int64, device=weights.device)
    return weights.index_select(0, index)


def _convert_sparse_conv_weight(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tuple(source.shape) == tuple(target.shape):
        return source
    if source.dim() != 5:
        raise ValueError(f"expected sparse conv donor weight to be 5D, got {tuple(source.shape)}")
    if target.dim() == 5:
        if source.shape[0] == target.shape[0] and source.shape[-1] == target.shape[1]:
            return source.permute(0, 4, 1, 2, 3).contiguous()
        if source.shape[-1] == target.shape[0] and source.shape[-2] == target.shape[1]:
            return source.permute(4, 3, 0, 1, 2).contiguous()
    elif target.dim() == 3:
        if source.shape[0] == target.shape[2] and source.shape[-1] == target.shape[1]:
            converted = source.permute(1, 2, 3, 4, 0).reshape(-1, target.shape[1], target.shape[2]).contiguous()
            return _reorder_k_offsets_if_needed(converted)
        if source.shape[-1] == target.shape[2] and source.shape[-2] == target.shape[1]:
            converted = source.reshape(-1, target.shape[1], target.shape[2]).contiguous()
            return _reorder_k_offsets_if_needed(converted)
    raise ValueError(f"cannot convert donor sparse conv weight {tuple(source.shape)} to target {tuple(target.shape)}")


def _extract_model_state(checkpoint: dict[str, object] | torch.Tensor) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if "model_state" in checkpoint and isinstance(checkpoint["model_state"], dict):
            return checkpoint["model_state"]
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
    raise TypeError("checkpoint does not contain a supported model state dict")


@dataclass(frozen=True)
class SecondCheckpointLoadReport:
    loaded_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    missing_target_keys: tuple[str, ...]


def _map_openpcdet_second_state(model: torch.nn.Module, donor_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_state = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    donor_sparse_conv_keys = [key for key in _donor_sparse_backbone_conv_keys() if key in donor_state]
    target_sparse_conv_keys = _target_sparse_backbone_conv_keys(model)
    if donor_sparse_conv_keys:
        if len(donor_sparse_conv_keys) != len(target_sparse_conv_keys):
            raise ValueError(
                f"sparse backbone conv key count mismatch: donor={len(donor_sparse_conv_keys)} target={len(target_sparse_conv_keys)}"
            )
        for donor_key, target_key in zip(donor_sparse_conv_keys, target_sparse_conv_keys):
            mapped[target_key] = _convert_sparse_conv_weight(donor_state[donor_key], target_state[target_key])
    donor_sparse_bn_keys = [key for key in _donor_sparse_backbone_bn_keys() if key in donor_state]
    target_sparse_bn_keys = _target_sparse_backbone_bn_keys(model)
    if donor_sparse_bn_keys and target_sparse_bn_keys:
        if len(donor_sparse_bn_keys) != len(target_sparse_bn_keys):
            raise ValueError(
                f"sparse backbone BN/stat key count mismatch: donor={len(donor_sparse_bn_keys)} target={len(target_sparse_bn_keys)}"
            )
        for donor_key, target_key in zip(donor_sparse_bn_keys, target_sparse_bn_keys):
            source_tensor = donor_state[donor_key]
            if tuple(source_tensor.shape) == tuple(target_state[target_key].shape):
                mapped[target_key] = source_tensor
    for donor_key, tensor in donor_state.items():
        if donor_key.startswith("backbone_3d."):
            continue
        target_key = donor_key.replace("backbone_2d.", "bev_backbone.", 1) if donor_key.startswith("backbone_2d.") else donor_key
        if target_key in target_state and tuple(tensor.shape) == tuple(target_state[target_key].shape):
            mapped[target_key] = tensor
    return mapped


def load_second_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, *, strict: bool = False) -> SecondCheckpointLoadReport:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    donor_state = _extract_model_state(checkpoint)
    target_state = model.state_dict()
    if all(key in target_state and tuple(value.shape) == tuple(target_state[key].shape) for key, value in donor_state.items()):
        mapped = {key: value for key, value in donor_state.items() if key in target_state}
    else:
        mapped = _map_openpcdet_second_state(model, donor_state)
    missing = tuple(sorted(key for key in target_state.keys() if key not in mapped))
    loaded = tuple(sorted(mapped.keys()))
    skipped = tuple(sorted(key for key in donor_state.keys() if key not in mapped and key not in loaded))
    model.load_state_dict(mapped, strict=strict)
    return SecondCheckpointLoadReport(loaded_keys=loaded, skipped_keys=skipped, missing_target_keys=missing)


@dataclass(frozen=True, slots=True)
class SparseBackboneOutput:
    encoded: object
    encoded_stride: int
    out_channels: int
    dense_shape_zyx: tuple[int, int, int]
    batch_size: int
    backend: str


def _conv_out_dim(size: int, kernel: int, stride: int, padding: int, dilation: int = 1) -> int:
    return int((size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1)


def second_encoded_dense_shape_zyx(input_shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    z, y, x = input_shape_zyx
    z = _conv_out_dim(z, 3, 2, 1)
    y = _conv_out_dim(y, 3, 2, 1)
    x = _conv_out_dim(x, 3, 2, 1)
    z = _conv_out_dim(z, 3, 2, 1)
    y = _conv_out_dim(y, 3, 2, 1)
    x = _conv_out_dim(x, 3, 2, 1)
    z = _conv_out_dim(z, 3, 2, 0)
    y = _conv_out_dim(y, 3, 2, 1)
    x = _conv_out_dim(x, 3, 2, 1)
    z = _conv_out_dim(z, 3, 2, 0)
    return z, y, x


def second_trunk_dense_shape_zyx(input_shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    z, y, x = input_shape_zyx
    z = _conv_out_dim(z, 3, 2, 1)
    y = _conv_out_dim(y, 3, 2, 1)
    x = _conv_out_dim(x, 3, 2, 1)
    z = _conv_out_dim(z, 3, 2, 1)
    y = _conv_out_dim(y, 3, 2, 1)
    x = _conv_out_dim(x, 3, 2, 1)
    z = _conv_out_dim(z, 3, 2, 0)
    y = _conv_out_dim(y, 3, 2, 1)
    x = _conv_out_dim(x, 3, 2, 1)
    return z, y, x


def _compress_dense_ncdhw(dense: torch.Tensor) -> torch.Tensor:
    if dense.dim() != 5:
        raise ValueError(f"expected dense sparse volume [N, C, D, H, W], got shape {tuple(dense.shape)}")
    n, c, d, h, w = dense.shape
    return dense.reshape(n, c * d, h, w).contiguous()


def _crop_or_pad_dense_ncdhw(dense: torch.Tensor, target_shape_zyx: tuple[int, int, int]) -> torch.Tensor:
    if dense.dim() != 5:
        raise ValueError(f"expected dense sparse volume [N, C, D, H, W], got shape {tuple(dense.shape)}")
    target_d, target_h, target_w = (int(v) for v in target_shape_zyx)
    out = dense.new_zeros((int(dense.size(0)), int(dense.size(1)), target_d, target_h, target_w))
    copy_d = min(int(dense.size(2)), target_d)
    copy_h = min(int(dense.size(3)), target_h)
    copy_w = min(int(dense.size(4)), target_w)
    out[:, :, :copy_d, :copy_h, :copy_w] = dense[:, :, :copy_d, :copy_h, :copy_w]
    return out


def dense_bev_from_backbone_output(output: SparseBackboneOutput) -> torch.Tensor:
    if output.backend == "minkowski":
        zero_min = torch.IntTensor([0, 0, 0])
        dense, _min_coord, _tensor_stride = output.encoded.dense(min_coordinate=zero_min)
        dense = _crop_or_pad_dense_ncdhw(dense, output.dense_shape_zyx)
        return _compress_dense_ncdhw(dense)
    if output.backend == "torchsparse":
        dense = output.encoded.dense()
        if dense.dim() != 5:
            raise ValueError(f"expected torchsparse dense output [N, D, H, W, C], got shape {tuple(dense.shape)}")
        n, d, h, w, c = dense.shape
        return dense.permute(0, 4, 1, 2, 3).reshape(n, c * d, h, w).contiguous()
    return _compress_dense_ncdhw(output.encoded.dense())


def available_kitti_second_backends() -> dict[str, dict[str, str | bool | None]]:
    result: dict[str, dict[str, str | bool | None]] = {
        "gtsparse": {"available": True, "error": None},
        "spconv": {"available": False, "error": None},
        "torchsparse": {"available": False, "error": None},
        "minkowski": {"available": False, "error": None},
    }
    try:
        import spconv.pytorch  # noqa: F401

        result["spconv"]["available"] = True
    except Exception as exc:
        result["spconv"]["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import torchsparse  # noqa: F401

        result["torchsparse"]["available"] = True
    except Exception as exc:
        result["torchsparse"]["error"] = f"{type(exc).__name__}: {exc}"
    try:
        import MinkowskiEngine  # noqa: F401

        result["minkowski"]["available"] = True
    except Exception as exc:
        result["minkowski"]["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _make_gtsparse_sparse_tensor(voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int, sparse_shape_zyx) -> GTSparseSparseConvTensor:
    return GTSparseSparseConvTensor(
        voxel_features,
        voxel_coords,
        sparse_shape_zyx,
        batch_size,
    )


class _GTPostActBlock(nn.Module):
    def __init__(self, conv: nn.Module, channels: int) -> None:
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        out = self.conv(x)
        out.replace_feature_(self.relu(self.bn(out.features)))
        return out


class _GTDownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, padding, sorted: bool = False) -> None:
        super().__init__()
        self.conv = GeometricTemplateSparseConv3d(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=padding,
            bias=False,
            build_reverse=False,
            sorted=bool(sorted),
        )
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.block1 = _GTPostActBlock(
            GeometricTemplateSubMConv3d(out_channels, out_channels, 3, padding=1, bias=False, sorted=bool(sorted)),
            out_channels,
        )
        self.block2 = _GTPostActBlock(
            GeometricTemplateSubMConv3d(out_channels, out_channels, 3, padding=1, bias=False, sorted=bool(sorted)),
            out_channels,
        )

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        x = self.conv(x)
        x.replace_feature_(self.relu(self.bn(x.features)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class MeanVoxelFeatureEncoder(nn.Module):
    def __init__(self, num_point_features: int) -> None:
        super().__init__()
        self.num_point_features = int(num_point_features)

    def forward(self, voxels: torch.Tensor, voxel_num_points: torch.Tensor) -> torch.Tensor:
        points_mean = voxels.sum(dim=1)
        normalizer = torch.clamp_min(voxel_num_points.view(-1, 1), min=1).type_as(voxels)
        return (points_mean / normalizer).contiguous()


class BaseBEVBackbone2D(nn.Module):
    def __init__(self, model_cfg: KittiSecondModelConfig) -> None:
        super().__init__()
        layer_nums = model_cfg.bev_layer_nums
        layer_strides = model_cfg.bev_layer_strides
        num_filters = model_cfg.bev_num_filters
        upsample_strides = model_cfg.bev_upsample_strides
        num_upsample_filters = model_cfg.bev_num_upsample_filters

        c_in_list = [model_cfg.num_bev_input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(len(layer_nums)):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(c_in_list[idx], num_filters[idx], kernel_size=3, stride=layer_strides[idx], padding=0, bias=False),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], upsample_strides[idx], stride=upsample_strides[idx], bias=False),
                nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
        self.num_bev_features = int(sum(num_upsample_filters))

    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        ups = []
        x = spatial_features
        for block, deblock in zip(self.blocks, self.deblocks):
            x = block(x)
            ups.append(deblock(x))
        if len(ups) > 1:
            min_h = min(int(u.size(2)) for u in ups)
            min_w = min(int(u.size(3)) for u in ups)
            ups = [u[:, :, :min_h, :min_w].contiguous() for u in ups]
        if len(ups) == 1:
            return ups[0]
        return torch.cat(ups, dim=1)


class SecondPredictionHead(nn.Module):
    def __init__(self, model_cfg: KittiSecondModelConfig, data_cfg: KittiSecondDataConfig, input_channels: int) -> None:
        super().__init__()
        num_class = len(data_cfg.class_names)
        num_anchors = model_cfg.num_anchors_per_location(data_cfg.class_names)
        self.num_class = int(num_class)
        self.box_code_size = int(model_cfg.box_code_size)
        self.num_dir_bins = int(model_cfg.num_dir_bins)
        self.conv_cls = nn.Conv2d(input_channels, num_anchors * num_class, kernel_size=1)
        self.conv_box = nn.Conv2d(input_channels, num_anchors * self.box_code_size, kernel_size=1)
        self.conv_dir_cls = nn.Conv2d(input_channels, num_anchors * self.num_dir_bins, kernel_size=1)

    def forward(self, spatial_features_2d: torch.Tensor) -> dict[str, torch.Tensor]:
        cls_preds = self.conv_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        box_preds = self.conv_box(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        dir_cls_preds = self.conv_dir_cls(spatial_features_2d).permute(0, 2, 3, 1).contiguous()
        return {
            "cls_preds": cls_preds,
            "box_preds": box_preds,
            "dir_cls_preds": dir_cls_preds,
        }


class GeometricTemplateVoxelBackBone8x(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: KittiSecondModelConfig, *, sorted: bool = False) -> None:
        super().__init__()
        self.model_name = "kitti_second_geometric_template"
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.sparse_shape_list = list(self.sparse_shape)
        self.trunk_shape = second_trunk_dense_shape_zyx(self.sparse_shape)
        self.out_shape = second_encoded_dense_shape_zyx(self.sparse_shape)
        self.conv_input = _GTPostActBlock(
            GeometricTemplateSubMConv3d(
                model_cfg.input_channels,
                model_cfg.conv_input_channels,
                3,
                padding=1,
                bias=False,
                sorted=bool(sorted),
            ),
            model_cfg.conv_input_channels,
        )
        self.conv1 = _GTPostActBlock(
            GeometricTemplateSubMConv3d(
                model_cfg.conv_input_channels,
                model_cfg.conv_input_channels,
                3,
                padding=1,
                bias=False,
                sorted=bool(sorted),
            ),
            model_cfg.conv_input_channels,
        )
        self.conv2 = _GTDownStage(model_cfg.conv_input_channels, model_cfg.conv2_channels, padding=1, sorted=bool(sorted))
        self.conv3 = _GTDownStage(model_cfg.conv2_channels, model_cfg.conv3_channels, padding=1, sorted=bool(sorted))
        self.conv4 = _GTDownStage(model_cfg.conv3_channels, model_cfg.conv4_channels, padding=(0, 1, 1), sorted=bool(sorted))
        self.conv_out = _GTPostActBlock(
            GeometricTemplateKernel3Conv3d(
                model_cfg.conv4_channels,
                model_cfg.conv_out_channels,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=0,
                bias=False,
            ),
            model_cfg.conv_out_channels,
        )
        self.out_channels = model_cfg.conv_out_channels

    def _forward_trunk_tensor(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> GTSparseSparseConvTensor:
        x = _make_gtsparse_sparse_tensor(voxel_features, voxel_coords, batch_size, self.sparse_shape_list)
        x = self.conv_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward_design_space_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> GTSparseSparseConvTensor:
        return self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)

    def forward_design_space(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self.forward_design_space_raw(voxel_features, voxel_coords, batch_size)
        return SparseBackboneOutput(
            encoded=x,
            encoded_stride=8,
            out_channels=x.features.size(1),
            dense_shape_zyx=self.trunk_shape,
            batch_size=batch_size,
            backend="gtsparse",
        )

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)
        x = self.conv_out(x)
        return SparseBackboneOutput(
            encoded=x,
            encoded_stride=8,
            out_channels=self.out_channels,
            dense_shape_zyx=self.out_shape,
            batch_size=batch_size,
            backend="gtsparse",
        )


class _TorchSparseTailBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, *, stride=1, padding=0, dilation=1, layer_name: str) -> None:
        super().__init__()
        import torchsparse
        import torchsparse.nn as spnn

        self._torchsparse = torchsparse
        self.layer_name = str(layer_name)
        self.block = nn.Sequential(
            spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            spnn.BatchNorm(out_channels, eps=1e-3, momentum=0.01),
            spnn.ReLU(),
        )

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        ts = self._torchsparse.SparseTensor(
            x.features,
            x.indices,
            spatial_range=(x.batch_size, *x.spatial_shape),
        )
        ts_out = self.block(ts)
        return x.replace_sparse(
            new_features=ts_out.feats,
            new_coords=ts_out.coords,
            new_spatial_shape=ts_out.spatial_range[1:],
            new_batch_size=ts_out.spatial_range[0],
            coord_hashmap=None,
            metadata=getattr(x, "metadata", None),
        )


class SpconvVoxelBackBone8x(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: KittiSecondModelConfig) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self._spconv = spconv
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.trunk_shape = second_trunk_dense_shape_zyx(self.sparse_shape)
        self.out_shape = second_encoded_dense_shape_zyx(self.sparse_shape)
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(model_cfg.input_channels, model_cfg.conv_input_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv_input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(model_cfg.conv_input_channels, model_cfg.conv_input_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv_input_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.conv2 = spconv.SparseSequential(
            spconv.SparseConv3d(model_cfg.conv_input_channels, model_cfg.conv2_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv2_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(model_cfg.conv2_channels, model_cfg.conv2_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv2_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(model_cfg.conv2_channels, model_cfg.conv2_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv2_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.conv3 = spconv.SparseSequential(
            spconv.SparseConv3d(model_cfg.conv2_channels, model_cfg.conv3_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv3_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(model_cfg.conv3_channels, model_cfg.conv3_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv3_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(model_cfg.conv3_channels, model_cfg.conv3_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv3_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.conv4 = spconv.SparseSequential(
            spconv.SparseConv3d(model_cfg.conv3_channels, model_cfg.conv4_channels, 3, stride=2, padding=(0, 1, 1), bias=False),
            nn.BatchNorm1d(model_cfg.conv4_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(model_cfg.conv4_channels, model_cfg.conv4_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv4_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            spconv.SubMConv3d(model_cfg.conv4_channels, model_cfg.conv4_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(model_cfg.conv4_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(model_cfg.conv4_channels, model_cfg.conv_out_channels, (3, 1, 1), stride=(2, 1, 1), padding=0, bias=False),
            nn.BatchNorm1d(model_cfg.conv_out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.out_channels = model_cfg.conv_out_channels

    def _forward_trunk_tensor(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._spconv.SparseConvTensor(voxel_features, voxel_coords.int(), list(self.sparse_shape), int(batch_size))
        x = self.conv_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward_design_space_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)

    def forward_design_space(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self.forward_design_space_raw(voxel_features, voxel_coords, batch_size)
        return SparseBackboneOutput(x, encoded_stride=8, out_channels=int(x.features.size(1)), dense_shape_zyx=self.trunk_shape, batch_size=int(batch_size), backend="spconv")

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)
        x = self.conv_out(x)
        return SparseBackboneOutput(x, encoded_stride=8, out_channels=self.out_channels, dense_shape_zyx=self.out_shape, batch_size=int(batch_size), backend="spconv")


class _TorchSparsePostActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, *, stride=1, padding=0, layer_name: str) -> None:
        super().__init__()
        import torchsparse.nn as spnn

        self.layer_name = str(layer_name)
        self.conv = spnn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = spnn.BatchNorm(out_channels, eps=1e-3, momentum=0.01)
        self.relu = spnn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class TorchSparseVoxelBackBone8x(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: KittiSecondModelConfig) -> None:
        super().__init__()
        import torchsparse

        self._torchsparse = torchsparse
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.trunk_shape = second_trunk_dense_shape_zyx(self.sparse_shape)
        self.out_shape = second_encoded_dense_shape_zyx(self.sparse_shape)
        self.conv_input = _TorchSparsePostActBlock(model_cfg.input_channels, model_cfg.conv_input_channels, 3, padding=1, layer_name="conv_input")
        self.conv1 = nn.Sequential(_TorchSparsePostActBlock(model_cfg.conv_input_channels, model_cfg.conv_input_channels, 3, padding=1, layer_name="conv1"))
        self.conv2 = nn.Sequential(
            _TorchSparsePostActBlock(model_cfg.conv_input_channels, model_cfg.conv2_channels, 3, stride=2, padding=1, layer_name="conv2.0"),
            _TorchSparsePostActBlock(model_cfg.conv2_channels, model_cfg.conv2_channels, 3, padding=1, layer_name="conv2.1"),
            _TorchSparsePostActBlock(model_cfg.conv2_channels, model_cfg.conv2_channels, 3, padding=1, layer_name="conv2.2"),
        )
        self.conv3 = nn.Sequential(
            _TorchSparsePostActBlock(model_cfg.conv2_channels, model_cfg.conv3_channels, 3, stride=2, padding=1, layer_name="conv3.0"),
            _TorchSparsePostActBlock(model_cfg.conv3_channels, model_cfg.conv3_channels, 3, padding=1, layer_name="conv3.1"),
            _TorchSparsePostActBlock(model_cfg.conv3_channels, model_cfg.conv3_channels, 3, padding=1, layer_name="conv3.2"),
        )
        self.conv4 = nn.Sequential(
            _TorchSparsePostActBlock(model_cfg.conv3_channels, model_cfg.conv4_channels, 3, stride=2, padding=(0, 1, 1), layer_name="conv4.0"),
            _TorchSparsePostActBlock(model_cfg.conv4_channels, model_cfg.conv4_channels, 3, padding=1, layer_name="conv4.1"),
            _TorchSparsePostActBlock(model_cfg.conv4_channels, model_cfg.conv4_channels, 3, padding=1, layer_name="conv4.2"),
        )
        self.conv_out = _TorchSparsePostActBlock(model_cfg.conv4_channels, model_cfg.conv_out_channels, (3, 1, 1), stride=(2, 1, 1), padding=0, layer_name="conv_out")
        self.out_channels = model_cfg.conv_out_channels

    def _forward_trunk_tensor(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        spatial_range = (int(batch_size), *self.sparse_shape)
        x = self._torchsparse.SparseTensor(voxel_features, voxel_coords, spatial_range=spatial_range)
        x = self.conv_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward_design_space_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)

    def forward_design_space(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self.forward_design_space_raw(voxel_features, voxel_coords, batch_size)
        return SparseBackboneOutput(x, encoded_stride=8, out_channels=int(x.feats.size(1)), dense_shape_zyx=self.trunk_shape, batch_size=int(batch_size), backend="torchsparse")

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)
        x = self.conv_out(x)
        return SparseBackboneOutput(x, encoded_stride=8, out_channels=self.out_channels, dense_shape_zyx=self.out_shape, batch_size=int(batch_size), backend="torchsparse")


class _MinkowskiPostActBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, *, stride=1, padding=0) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self.block = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dimension=3),
            ME.MinkowskiBatchNorm(out_channels, eps=1e-3, momentum=0.01),
            ME.MinkowskiReLU(),
        )

    def forward(self, x):
        return self.block(x)


class MinkowskiVoxelBackBone8x(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: KittiSecondModelConfig) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self._ME = ME
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.trunk_shape = second_trunk_dense_shape_zyx(self.sparse_shape)
        self.out_shape = second_encoded_dense_shape_zyx(self.sparse_shape)
        self.conv_input = nn.Sequential(
            ME.MinkowskiConvolution(model_cfg.input_channels, model_cfg.conv_input_channels, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(model_cfg.conv_input_channels, eps=1e-3, momentum=0.01),
            ME.MinkowskiReLU(),
        )
        self.conv1 = nn.Sequential(_MinkowskiPostActBlock(model_cfg.conv_input_channels, model_cfg.conv_input_channels, 3))
        self.conv2 = nn.Sequential(
            _MinkowskiPostActBlock(model_cfg.conv_input_channels, model_cfg.conv2_channels, 3, stride=2),
            _MinkowskiPostActBlock(model_cfg.conv2_channels, model_cfg.conv2_channels, 3),
            _MinkowskiPostActBlock(model_cfg.conv2_channels, model_cfg.conv2_channels, 3),
        )
        self.conv3 = nn.Sequential(
            _MinkowskiPostActBlock(model_cfg.conv2_channels, model_cfg.conv3_channels, 3, stride=2),
            _MinkowskiPostActBlock(model_cfg.conv3_channels, model_cfg.conv3_channels, 3),
            _MinkowskiPostActBlock(model_cfg.conv3_channels, model_cfg.conv3_channels, 3),
        )
        self.conv4 = nn.Sequential(
            _MinkowskiPostActBlock(model_cfg.conv3_channels, model_cfg.conv4_channels, 3, stride=2),
            _MinkowskiPostActBlock(model_cfg.conv4_channels, model_cfg.conv4_channels, 3),
            _MinkowskiPostActBlock(model_cfg.conv4_channels, model_cfg.conv4_channels, 3),
        )
        self.conv_out = nn.Sequential(
            ME.MinkowskiConvolution(model_cfg.conv4_channels, model_cfg.conv_out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), dimension=3),
            ME.MinkowskiBatchNorm(model_cfg.conv_out_channels, eps=1e-3, momentum=0.01),
            ME.MinkowskiReLU(),
        )
        self.out_channels = model_cfg.conv_out_channels

    def _forward_trunk_tensor(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._ME.SparseTensor(voxel_features, coordinates=voxel_coords)
        x = self.conv_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward_design_space_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)

    def forward_design_space(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self.forward_design_space_raw(voxel_features, voxel_coords, batch_size)
        return SparseBackboneOutput(x, encoded_stride=8, out_channels=int(x.features.size(1)), dense_shape_zyx=self.trunk_shape, batch_size=int(batch_size), backend="minkowski")

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBackboneOutput:
        x = self._forward_trunk_tensor(voxel_features, voxel_coords, batch_size)
        x = self.conv_out(x)
        return SparseBackboneOutput(x, encoded_stride=8, out_channels=self.out_channels, dense_shape_zyx=self.out_shape, batch_size=int(batch_size), backend="minkowski")


def create_sparse_backbone(
    backend: str,
    *,
    data_cfg: KittiSecondDataConfig,
    model_cfg: KittiSecondModelConfig,
    sorted: bool = False,
):
    normalized = str(backend).lower()
    if normalized == "gtsparse":
        return GeometricTemplateVoxelBackBone8x(data_cfg, model_cfg, sorted=bool(sorted))
    if normalized == "spconv":
        return SpconvVoxelBackBone8x(data_cfg, model_cfg)
    if normalized == "torchsparse":
        return TorchSparseVoxelBackBone8x(data_cfg, model_cfg)
    if normalized == "minkowski":
        return MinkowskiVoxelBackBone8x(data_cfg, model_cfg)
    raise KeyError(f"unsupported backend {backend!r}")


class KittiSecondModel(nn.Module):
    def __init__(self, *, backend: str, config: KittiSecondConfig | None = None, sorted: bool = False) -> None:
        super().__init__()
        self.config = config if config is not None else KittiSecondConfig()
        self.backend = str(backend).lower()
        self.sorted = bool(sorted)
        if self.backend in {"gtsparse", "torchsparse"}:
            _configure_torchsparse_conv_sort_mode(self.sorted)
        self.vfe = MeanVoxelFeatureEncoder(self.config.data.num_point_features)
        self.sparse_backbone = create_sparse_backbone(
            self.backend,
            data_cfg=self.config.data,
            model_cfg=self.config.model,
            sorted=self.sorted,
        )
        self.bev_backbone = BaseBEVBackbone2D(self.config.model)
        self.dense_head = SecondPredictionHead(self.config.model, self.config.data, input_channels=self.bev_backbone.num_bev_features)

    @property
    def model_config_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "sorted": self.sorted,
            "data": asdict(self.config.data),
            "model": asdict(self.config.model),
        }

    def _extract_batch_tensors(
        self,
        batch: KittiSecondBatch | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if isinstance(batch, KittiSecondBatch):
            return batch.voxels, batch.voxel_coords, batch.voxel_num_points, int(batch.batch_size)
        return batch["voxels"], batch["voxel_coords"], batch["voxel_num_points"], int(batch["batch_size"])

    def encode_batch(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        voxels, voxel_coords, voxel_num_points, batch_size = self._extract_batch_tensors(batch)
        voxel_features = self.vfe(voxels, voxel_num_points)
        return voxel_features, voxel_coords, batch_size

    def forward_sparse_backbone(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self.sparse_backbone(voxel_features, voxel_coords, int(batch_size))

    def forward_sparse_convolutions(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self.sparse_backbone(voxel_features, voxel_coords, int(batch_size))

    def forward_sparse_design_space(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self.sparse_backbone.forward_design_space(voxel_features, voxel_coords, int(batch_size))

    def forward_sparse_design_space_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self.sparse_backbone.forward_design_space_raw(voxel_features, voxel_coords, int(batch_size))

    def forward_sparse_backbone_from_batch(self, batch: KittiSecondBatch | dict[str, torch.Tensor]):
        voxel_features, voxel_coords, batch_size = self.encode_batch(batch)
        return self.forward_sparse_backbone(voxel_features, voxel_coords, batch_size)

    def forward_dense_bev_from_sparse_output(self, sparse_output) -> torch.Tensor:
        return dense_bev_from_backbone_output(sparse_output)

    def forward_dense_bev_from_batch(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.forward_dense_bev_from_sparse_output(self.forward_sparse_backbone_from_batch(batch))

    def forward(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sparse_output = self.forward_sparse_backbone_from_batch(batch)
        spatial_features = self.forward_dense_bev_from_sparse_output(sparse_output)
        spatial_features_2d = self.bev_backbone(spatial_features)
        predictions = self.dense_head(spatial_features_2d)
        predictions["spatial_features"] = spatial_features
        predictions["spatial_features_2d"] = spatial_features_2d
        return predictions

    def decode_predictions(self, predictions: dict[str, torch.Tensor]) -> DecodedSecondPredictions:
        return decode_second_predictions(predictions, data_cfg=self.config.data, model_cfg=self.config.model)

    def postprocess_topk(self, predictions: dict[str, torch.Tensor], *, topk: int = 100) -> list[dict[str, torch.Tensor]]:
        return select_topk_predictions(self.decode_predictions(predictions), topk=topk)

    def postprocess_nms(
        self,
        predictions: dict[str, torch.Tensor],
        *,
        score_thresh: float = 0.1,
        nms_thresh: float = 0.01,
        pre_maxsize: int = 4096,
        post_maxsize: int = 500,
    ) -> list[dict[str, torch.Tensor]]:
        return select_nms_predictions(
            self.decode_predictions(predictions),
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            pre_maxsize=pre_maxsize,
            post_maxsize=post_maxsize,
        )

    def load_checkpoint(self, checkpoint_path: str | Path, *, strict: bool = False) -> SecondCheckpointLoadReport:
        return load_second_checkpoint(self, checkpoint_path, strict=strict)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KITTI + SECOND thin end-to-end runner")
    parser.add_argument("--backend", type=str, default="gtsparse", help="Sparse backend to run: gtsparse/spconv/torchsparse/minkowski")
    parser.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16"))
    parser.add_argument("--sorted", action="store_true", help="Enable sorted sparse-conv variants for GTSparse and TorchSparse backends.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Optional SECOND checkpoint")
    parser.add_argument("--strict-ckpt", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/kitti"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--frame", type=str, default="")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--score-thresh", type=float, default=0.1)
    parser.add_argument("--nms-thresh", type=float, default=0.01)
    parser.add_argument("--post-maxsize", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=5)
    parser.add_argument("--timing-warmup-repeats", type=int, default=2)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Root directory under which auto-generated run logs are written.")
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _iter_sample_indices(
    dataset: KittiLidarDataset,
    *,
    frame_id: str,
    num_samples: int,
    random_sample: bool = False,
) -> list[int]:
    if frame_id:
        try:
            return [dataset.sample_ids.index(frame_id)]
        except ValueError as exc:
            raise KeyError(f"frame_id {frame_id!r} is not present in split {dataset.split}") from exc
    if int(num_samples) <= 0:
        count = len(dataset)
    else:
        count = min(len(dataset), int(num_samples))
    if random_sample and count < len(dataset):
        return sorted(random.Random(0).sample(range(len(dataset)), count))
    return list(range(count))


def _make_loader(dataset: KittiLidarDataset, indices: list[int], data_cfg: KittiSecondDataConfig, *, batch_size: int) -> torch_data.DataLoader:
    subset = torch_data.Subset(dataset, indices)
    return torch_data.DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=2,
        collate_fn=lambda samples: collate_kitti_second_batch(samples, data_cfg=data_cfg, training=False),
    )


def _iter_device_batches(loader: torch_data.DataLoader, device: str, *, dtype: torch.dtype):
    for batch in loader:
        yield move_kitti_second_batch_to_device(batch, device, dtype=dtype)


def _measure_frame_timings(
    model: KittiSecondModel,
    loader: torch_data.DataLoader,
    *,
    device: str,
    warmup: int,
    timing_repeats: int,
    timing_warmup_repeats: int,
    topk: int,
    score_thresh: float,
    nms_thresh: float,
    post_maxsize: int,
    progress_desc: str = "sparse_inference",
    on_result=None,
):
    local_measure_warmup_repeats = max(0, int(timing_warmup_repeats))
    local_measure_repeats = max(1, int(timing_repeats))
    resolved_device = require_cuda_device(device)
    runtime_dtype = next(model.parameters()).dtype
    conv_only_fn = model.forward_sparse_convolutions
    clear_metadata = lambda: clear_sparse_metadata(model.backend)

    warmup_device_batches = _iter_device_batches(loader, device, dtype=runtime_dtype)
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            batch = next(warmup_device_batches, None)
            if batch is None:
                break
            model(batch)
            torch.cuda.synchronize(device=resolved_device)
            clear_metadata()

    results = []
    device_batches = _iter_device_batches(loader, device, dtype=runtime_dtype)
    measured_batches = tqdm(device_batches, total=len(loader), desc=f"{progress_desc}/full", dynamic_ncols=True)
    with torch.no_grad():
        for batch_index, batch in enumerate(measured_batches):
            predictions, end2end_ms = measure_cuda_elapsed_ms(
                model,
                batch,
                device=resolved_device,
                repeats=local_measure_repeats,
                warmup_repeats=local_measure_warmup_repeats,
                clear_metadata=clear_metadata,
            )
            topk_preds = model.postprocess_topk(predictions, topk=int(topk))
            nms_preds = model.postprocess_nms(
                predictions,
                score_thresh=float(score_thresh),
                nms_thresh=float(nms_thresh),
                post_maxsize=int(post_maxsize),
            )
            record = {
                "batch_index": int(batch_index),
                "frame_ids": list(batch.frame_ids),
                "end2end_ms": float(end2end_ms),
                "topk_count": int(topk_preds[0]["pred_scores"].numel()) if topk_preds else 0,
                "nms_count": int(nms_preds[0]["pred_scores"].numel()) if nms_preds else 0,
                "timing_repeats": int(local_measure_repeats),
                "timing_warmup_repeats": int(local_measure_warmup_repeats),
            }
            results.append(record)
            del predictions, topk_preds, nms_preds

    warmup_device_batches = _iter_device_batches(loader, device, dtype=runtime_dtype)
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            batch = next(warmup_device_batches, None)
            if batch is None:
                break
            voxel_features, voxel_coords, batch_size = model.encode_batch(batch)
            conv_only_fn(voxel_features, voxel_coords, batch_size)
            torch.cuda.synchronize(device=resolved_device)
            clear_metadata()

    device_batches = _iter_device_batches(loader, device, dtype=runtime_dtype)
    measured_batches = tqdm(device_batches, total=len(loader), desc=f"{progress_desc}/conv", dynamic_ncols=True)
    with torch.no_grad():
        for batch_index, batch in enumerate(measured_batches):
            record = results[batch_index]
            if list(batch.frame_ids) != record["frame_ids"]:
                raise RuntimeError("full and conv-only passes visited different frames")
            voxel_features, voxel_coords, batch_size = model.encode_batch(batch)
            sparse_output, conv_only_ms = measure_cuda_elapsed_ms(
                conv_only_fn,
                voxel_features,
                voxel_coords,
                batch_size,
                device=resolved_device,
                repeats=local_measure_repeats,
                warmup_repeats=local_measure_warmup_repeats,
                clear_metadata=clear_metadata,
            )
            record["conv_only_ms"] = float(conv_only_ms)
            record["encoded_stride"] = int(getattr(sparse_output, "encoded_stride", getattr(model.config.model, "anchor_feature_map_stride", 8)))
            if on_result is not None:
                on_result(record)
            del sparse_output
    return results


def _sanitize_path_component(value: str) -> str:
    sanitized = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_", "."}:
            sanitized.append(ch)
        else:
            sanitized.append("_")
    result = "".join(sanitized).strip("_")
    return result or "unknown"


def _device_index(device: str) -> int:
    parsed = torch.device(device)
    if parsed.index is not None:
        return int(parsed.index)
    return int(torch.cuda.current_device())


def _gpu_model_label(device: str) -> str:
    require_cuda_device(device)
    return _sanitize_path_component(torch.cuda.get_device_name(_device_index(device)))


def _dtype_label(model: KittiSecondModel) -> str:
    return _sanitize_path_component(str(next(model.parameters()).dtype).replace("torch.", ""))


def _log_dir_for_run(*, root_log_dir: Path, device: str, model: KittiSecondModel, data_root: Path, sweeps: int) -> Path:
    gpu_model = _gpu_model_label(device)
    dtype_name = _dtype_label(model)
    model_name = "second"
    dataset_name = _sanitize_path_component(data_root.name)
    return Path(root_log_dir) / f"logs_{gpu_model}_{dtype_name}_{model_name}_{dataset_name}_sweeps{int(sweeps)}"


def _append_backend_log_frame(log_file, record: dict[str, object]) -> None:
    payload = {
        "batch_index": int(record["batch_index"]),
        "frame_ids": list(record["frame_ids"]),
        "conv_only_ms": float(record["conv_only_ms"]),
        "end2end_ms": float(record["end2end_ms"]),
        "timing_repeats": int(record.get("timing_repeats", 1)),
        "timing_warmup_repeats": int(record.get("timing_warmup_repeats", 0)),
    }
    json.dump(payload, log_file, ensure_ascii=True, sort_keys=True)
    log_file.write("\n")
    log_file.flush()


def _stats_dict(values: list[float]) -> dict[str, float | int] | None:
    finite_values = [float(v) for v in values if math.isfinite(float(v))]
    if not finite_values:
        return None
    return {
        "count": int(len(finite_values)),
        "median_ms": float(statistics.median(finite_values)),
        "mean_ms": float(statistics.mean(finite_values)),
        "min_ms": float(min(finite_values)),
        "max_ms": float(max(finite_values)),
    }


def _write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def _format_stats(name: str, values: list[float]) -> str:
    stats = _stats_dict(values)
    if stats is None:
        return f"{name}: n/a"
    return (
        f"{name}: median={float(stats['median_ms']):.3f}ms "
        f"mean={float(stats['mean_ms']):.3f}ms min={float(stats['min_ms']):.3f}ms max={float(stats['max_ms']):.3f}ms"
    )


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    runtime_dtype = resolve_runtime_dtype(args.dtype)
    require_cuda_device(args.device)
    requested_sweeps = max(1, int(getattr(args, "sweeps", 1)))
    if requested_sweeps != 1:
        raise ValueError("KITTI + SECOND thin runner currently supports only --sweeps 1; this dataset layout does not provide aligned multi-sweep poses")
    if str(args.backend).lower() == "minkowski" and runtime_dtype == torch.float16:
        raise ValueError("Minkowski backend does not support fp16 in this runner")
    config = KittiSecondConfig(
        data=KittiSecondDataConfig(
            root=discover_kitti_root(args.data_root),
            split=str(args.split),
            max_sweeps=requested_sweeps,
        )
    )
    model = KittiSecondModel(backend=args.backend, config=config, sorted=bool(getattr(args, "sorted", False))).to(args.device)
    if runtime_dtype == torch.float16:
        model = model.half()
    model.eval()
    checkpoint_report = None
    if args.ckpt is not None:
        checkpoint_report = model.load_checkpoint(args.ckpt, strict=bool(args.strict_ckpt))

    dataset = KittiLidarDataset(config.data.root, split=config.data.split)
    indices = _iter_sample_indices(dataset, frame_id=str(args.frame), num_samples=int(args.frames))
    loader = _make_loader(dataset, indices, config.data, batch_size=int(args.batch))
    warmup_batches = min(int(args.warmup), len(loader))
    measured_batches = max(0, len(loader) - warmup_batches)
    root_log_dir = Path(getattr(args, "log_dir", Path("logs")))
    log_dir = _log_dir_for_run(root_log_dir=root_log_dir, device=str(args.device), model=model, data_root=config.data.root, sweeps=int(config.data.max_sweeps))
    log_path = log_dir / f"{args.backend}.jsonl"
    config_path = log_dir / f"{args.backend}.config.json"
    summary_path = log_dir / f"{args.backend}.summary.json"
    gpu_name = torch.cuda.get_device_name(_device_index(str(args.device)))
    workload = "second_kitti_sweeps1"
    _write_json_file(
        config_path,
        {
            "backend": str(args.backend),
            "batch": int(args.batch),
            "ckpt": None if args.ckpt is None else str(args.ckpt),
            "data_root": str(config.data.root),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "sorted": bool(getattr(args, "sorted", False)),
            "frame": str(args.frame),
            "frames": int(args.frames),
            "gpu": gpu_name,
            "log_dir": str(log_dir),
            "nms_thresh": float(args.nms_thresh),
            "post_maxsize": int(args.post_maxsize),
            "score_thresh": float(args.score_thresh),
            "spconv_do_sort": True,
            "split": str(args.split),
            "strict_ckpt": bool(args.strict_ckpt),
            "sweeps": int(config.data.max_sweeps),
            "timing_repeats": int(max(1, args.timing_repeats)),
            "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
            "topk": int(args.topk),
            "warmup": int(args.warmup),
            "workload": workload,
        },
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        results = _measure_frame_timings(
            model,
            loader,
            device=str(args.device),
            warmup=warmup_batches,
            timing_repeats=int(args.timing_repeats),
            timing_warmup_repeats=int(args.timing_warmup_repeats),
            topk=int(args.topk),
            score_thresh=float(args.score_thresh),
            nms_thresh=float(args.nms_thresh),
            post_maxsize=int(args.post_maxsize),
            progress_desc=f"second/kitti/{args.backend}",
            on_result=lambda record: _append_backend_log_frame(log_file, record),
        )
    conv_only_times = [float(record["conv_only_ms"]) for record in results]
    end2end_times = [float(record["end2end_ms"]) for record in results]

    summary = {
        "backend": str(args.backend),
        "dtype": str(args.dtype),
        "device": str(args.device),
        "gpu": gpu_name,
        "data_root": str(config.data.root),
        "split": str(args.split),
        "sweeps": int(config.data.max_sweeps),
        "frames_requested": int(len(indices)),
        "frames": int(len(results)),
        "warmup_batches": int(warmup_batches),
        "timing_repeats": int(max(1, args.timing_repeats)),
        "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
        "batch": int(args.batch),
        "log_dir": str(log_dir),
        "log_jsonl": str(log_path),
        "config_json": str(config_path),
        "summary_json": str(summary_path),
        "workload": workload,
        "results": results,
    }
    if checkpoint_report is not None:
        summary["checkpoint"] = {
            "loaded_keys": list(checkpoint_report.loaded_keys),
            "skipped_keys": list(checkpoint_report.skipped_keys),
            "missing_target_keys": list(checkpoint_report.missing_target_keys),
        }
    _write_json_file(
        summary_path,
        {
            "backend": str(args.backend),
            "batch": int(args.batch),
            "data_root": str(config.data.root),
            "dtype": str(args.dtype),
            "frames_logged": int(len(results)),
            "gpu": gpu_name,
            "split": str(args.split),
            "spconv_do_sort": True,
            "sweeps": int(config.data.max_sweeps),
            "stats": {
                "conv_only": _stats_dict(conv_only_times),
                "end2end": _stats_dict(end2end_times),
            },
            "timing_repeats": int(max(1, args.timing_repeats)),
            "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
            "warmup_batches": int(warmup_batches),
            "workload": workload,
        },
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = _parse_args()
    summary = run_cli(args)
    print("=" * 60)
    print(f"backend={summary['backend']}")
    print(f"data_root={summary['data_root']}")
    print(f"dtype={summary['dtype']}")
    print(f"split={summary['split']}")
    print(f"frames={summary['frames']} warmup={summary['warmup_batches']} requested={summary['frames_requested']} batch={summary['batch']}")
    print(f"log_jsonl={summary['log_jsonl']}")
    print(f"config_json={summary['config_json']}")
    print(f"summary_json={summary['summary_json']}")
    conv_only_times = [float(record['conv_only_ms']) for record in summary['results']]
    end2end_times = [float(record['end2end_ms']) for record in summary['results']]
    if conv_only_times:
        print(_format_stats("conv_only", conv_only_times))
    if end2end_times:
        print(_format_stats("end2end", end2end_times))
    print("=" * 60)


if __name__ == "__main__":
    main()
