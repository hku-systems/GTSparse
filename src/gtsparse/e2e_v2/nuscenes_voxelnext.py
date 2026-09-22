"""nuScenes + VoxelNeXt thin runner using KITTI input during bring-up."""

from __future__ import annotations

import argparse
from functools import reduce
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data


import torchsparse

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torchsparse.backends.allow_tf32 = False

from gtsparse.e2e_v2.common import require_cuda_device, resolve_runtime_dtype
from gtsparse.sparse3d.geometric_template import (
    GeometricTemplateKernel9Conv3d,
    GeometricTemplateSparseConv3d,
    GeometricTemplateSubMConv3d,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

from .kitti_second import (
    KittiLidarDataset,
    KittiSecondBatch,
    KittiSecondDataConfig,
    MeanVoxelFeatureEncoder,
    DecodedSecondPredictions,
    _configure_torchsparse_conv_sort_mode,
    _append_backend_log_frame,
    _format_stats,
    _iter_sample_indices,
    _make_loader,
    _measure_frame_timings,
    _sanitize_path_component,
    _stats_dict,
    _write_json_file,
    decode_second_predictions,
    discover_kitti_root,
    select_nms_predictions,
    select_topk_predictions,
)


@dataclass(frozen=True)
class NuScenesVoxelNeXtModelConfig:
    input_channels: int = 4
    channels: tuple[int, int, int, int, int] = (16, 32, 64, 128, 128)
    out_channels: int = 128
    box_code_size: int = 7
    num_dir_bins: int = 2
    anchor_rotations: tuple[float, ...] = (0.0, 1.57)
    anchor_sizes: tuple[tuple[float, float, float], ...] = (
        (3.9, 1.6, 1.56),
        (0.8, 0.6, 1.73),
        (1.76, 0.6, 1.73),
    )
    anchor_bottom_heights: tuple[float, ...] = (-1.78, -0.6, -0.6)
    anchor_feature_map_stride: int = 8

    def num_anchors_per_location(self, class_names: Sequence[str]) -> int:
        return int(len(class_names) * len(self.anchor_rotations))


@dataclass(frozen=True)
class NuScenesVoxelNeXtConfig:
    data: KittiSecondDataConfig = field(default_factory=lambda: NuScenesVoxelNeXtDataConfig())
    model: NuScenesVoxelNeXtModelConfig = field(default_factory=NuScenesVoxelNeXtModelConfig)


@dataclass(frozen=True)
class NuScenesVoxelNeXtDataConfig(KittiSecondDataConfig):
    root: Path = Path("dataset/nuscenes")
    split: str = "test"
    class_names: tuple[str, ...] = (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    )
    point_cloud_range: tuple[float, float, float, float, float, float] = (-54.0, -54.0, -5.0, 54.0, 54.0, 3.0)
    voxel_size: tuple[float, float, float] = (0.075, 0.075, 0.2)
    num_point_features: int = 4
    max_points_per_voxel: int = 10
    max_voxels_train: int = 120000
    max_voxels_eval: int = 160000
    max_sweeps: int = 10


@dataclass(frozen=True, slots=True)
class SparseBEVOutput:
    bev_features: torch.Tensor
    bev_coords: torch.Tensor
    spatial_shape_hw: tuple[int, int]
    encoded_stride: int
    batch_size: int
    backend: str


@dataclass(frozen=True, slots=True)
class SparseTrunkOutput:
    features: torch.Tensor
    coords_bzyx: torch.Tensor
    spatial_shape_zyx: tuple[int, int, int]
    encoded_stride: int
    batch_size: int
    backend: str


def available_nuscenes_voxelnext_backends() -> dict[str, dict[str, str | bool | None]]:
    from .kitti_second import available_kitti_second_backends

    return available_kitti_second_backends()


def discover_nuscenes_root(root: str | Path | None = None) -> Path:
    if root is None:
        return Path("dataset/nuscenes")
    return Path(root)


def _resolve_nuscenes_dataroot_and_version(root: str | Path) -> tuple[Path, str]:
    base = discover_nuscenes_root(root)
    if (base / "v1.0-trainval").exists():
        return base, "v1.0-trainval"
    if (base / "v1.0-mini").exists():
        return base, "v1.0-mini"
    if (base / "v1.0-test").exists():
        return base, "v1.0-test"
    if base.name.startswith("v1.0-") and base.exists():
        return base.parent, base.name
    raise FileNotFoundError(f"Could not find nuScenes version directory under {base}. Expected v1.0-mini, v1.0-trainval, or v1.0-test.")


def _resolve_nuscenes_split_scene_names(version_name: str, split: str) -> list[str]:
    from nuscenes.utils.splits import create_splits_scenes

    split_name = str(split).lower()
    if version_name == "v1.0-mini":
        mapping = {
            "train": "mini_train",
            "mini_train": "mini_train",
            "val": "mini_val",
            "mini_val": "mini_val",
            "test": "mini_val",
        }
    elif version_name == "v1.0-test":
        mapping = {
            "train": "test",
            "val": "test",
            "test": "test",
        }
    else:
        mapping = {
            "train": "train",
            "val": "val",
            "test": "test",
        }
    try:
        split_key = mapping[split_name]
    except KeyError as exc:
        raise KeyError(f"unsupported nuScenes split {split!r} for version {version_name!r}") from exc
    return list(create_splits_scenes()[split_key])


def _read_nuscenes_points_xyzi(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.float32)
    if raw.size % 5 == 0:
        points = raw.reshape(-1, 5)
    elif raw.size % 4 == 0:
        points = raw.reshape(-1, 4)
    else:
        raise ValueError(f"nuScenes lidar file {path} does not contain a multiple of 4 or 5 float32 values")
    if points.shape[1] < 4:
        raise ValueError(f"nuScenes lidar file {path} has fewer than 4 point features")
    return np.ascontiguousarray(points[:, :4], dtype=np.float32)


def _transform_nuscenes_points(points_xyzi: np.ndarray, transform_matrix: np.ndarray | None) -> np.ndarray:
    if transform_matrix is None:
        return np.ascontiguousarray(points_xyzi, dtype=np.float32)
    xyz1 = np.concatenate((points_xyzi[:, :3], np.ones((int(points_xyzi.shape[0]), 1), dtype=np.float32)), axis=1)
    transformed_xyz = (np.asarray(transform_matrix, dtype=np.float32) @ xyz1.T).T[:, :3]
    return np.ascontiguousarray(np.concatenate((transformed_xyz, points_xyzi[:, 3:4]), axis=1), dtype=np.float32)


def _remove_close_nuscenes_points(points_xyzi: np.ndarray, radius: float = 1.0) -> np.ndarray:
    x_filt = np.abs(points_xyzi[:, 0]) < float(radius)
    y_filt = np.abs(points_xyzi[:, 1]) < float(radius)
    mask = ~(x_filt & y_filt)
    return np.ascontiguousarray(points_xyzi[mask], dtype=np.float32)


class NuScenesLidarDataset(torch_data.Dataset):
    def __init__(self, root: str | Path, *, split: str = "val", max_sweeps: int = 10) -> None:
        super().__init__()
        from nuscenes.nuscenes import NuScenes

        self.root = discover_nuscenes_root(root)
        self.dataroot, self.version_name = _resolve_nuscenes_dataroot_and_version(self.root)
        self.split = str(split)
        self.max_sweeps = max(1, int(max_sweeps))
        self.nusc = NuScenes(version=self.version_name, dataroot=str(self.dataroot), verbose=False)
        target_scene_names = set(_resolve_nuscenes_split_scene_names(self.version_name, self.split))
        self.sample_ids: list[str] = []
        self.records: list[dict[str, Any]] = []
        for scene in self.nusc.scene:
            if str(scene["name"]) not in target_scene_names:
                continue
            token = str(scene["first_sample_token"])
            while token:
                sample = self.nusc.get("sample", token)
                self.sample_ids.append(token)
                self.records.append(self._build_sample_record(sample))
                token = str(sample["next"]) if sample["next"] else ""
        if not self.sample_ids:
            raise RuntimeError(f"No nuScenes samples found for split {self.split!r} under {self.dataroot / self.version_name}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _build_sample_record(self, sample_rec: dict[str, Any]) -> dict[str, Any]:
        from nuscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        ref_sd_rec = self.nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        ref_pose_rec = self.nusc.get("ego_pose", ref_sd_rec["ego_pose_token"])
        ref_cs_rec = self.nusc.get("calibrated_sensor", ref_sd_rec["calibrated_sensor_token"])
        ref_time = 1e-6 * float(ref_sd_rec["timestamp"])
        ref_from_car = transform_matrix(ref_cs_rec["translation"], Quaternion(ref_cs_rec["rotation"]), inverse=True)
        car_from_global = transform_matrix(ref_pose_rec["translation"], Quaternion(ref_pose_rec["rotation"]), inverse=True)
        sweeps: list[dict[str, Any]] = []
        current_sd_rec = ref_sd_rec
        for _ in range(self.max_sweeps):
            current_pose_rec = self.nusc.get("ego_pose", current_sd_rec["ego_pose_token"])
            current_cs_rec = self.nusc.get("calibrated_sensor", current_sd_rec["calibrated_sensor_token"])
            global_from_car = transform_matrix(current_pose_rec["translation"], Quaternion(current_pose_rec["rotation"]), inverse=False)
            car_from_current = transform_matrix(current_cs_rec["translation"], Quaternion(current_cs_rec["rotation"]), inverse=False)
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current]).astype(np.float32)
            sweeps.append(
                {
                    "path": self.dataroot / str(current_sd_rec["filename"]),
                    "transform": trans_matrix,
                    "time_lag": ref_time - 1e-6 * float(current_sd_rec["timestamp"]),
                }
            )
            prev_token = str(current_sd_rec["prev"])
            if prev_token == "":
                break
            current_sd_rec = self.nusc.get("sample_data", prev_token)
        return {
            "frame_id": str(sample_rec["token"]),
            "timestamp": float(sample_rec.get("timestamp", 0.0)),
            "token": str(sample_rec["token"]),
            "sweeps": sweeps,
        }

    def _load_sample_points(self, record: dict[str, Any]) -> np.ndarray:
        points = []
        for sweep in record["sweeps"]:
            sweep_points = _read_nuscenes_points_xyzi(Path(sweep["path"]))
            sweep_points = _transform_nuscenes_points(sweep_points, sweep["transform"])
            sweep_points = _remove_close_nuscenes_points(sweep_points, radius=1.0)
            points.append(sweep_points)
        if not points:
            return np.empty((0, 4), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(points, axis=0), dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {
            "frame_id": str(record["frame_id"]),
            "points": self._load_sample_points(record),
            "timestamp": float(record["timestamp"]),
            "token": str(record["token"]),
        }


def _make_nuscenes_dataset(data_cfg: KittiSecondDataConfig):
    root = Path(data_cfg.root)
    if (root / "v1.0-mini").exists() or (root / "v1.0-trainval").exists() or (root / "v1.0-test").exists() or root.name.startswith("v1.0-"):
        max_sweeps = int(getattr(data_cfg, "max_sweeps", 1))
        return NuScenesLidarDataset(root, split=str(data_cfg.split), max_sweeps=max_sweeps)
    return KittiLidarDataset(root, split=str(data_cfg.split))


def _configure_torchsparse_backend_for_nuscenes() -> int:
    import torchsparse.backends

    torchsparse.backends.hash_rsv_ratio = max(int(torchsparse.backends.hash_rsv_ratio), 4)
    return int(torchsparse.backends.hash_rsv_ratio)


def _make_gtsparse_sparse_tensor(voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int, sparse_shape_zyx) -> GTSparseSparseConvTensor:
    return GTSparseSparseConvTensor(voxel_features, voxel_coords, sparse_shape_zyx, batch_size)


def _collapse_to_bev_sparse(features: torch.Tensor, coords: torch.Tensor, spatial_shape_zyx: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    if int(coords.size(1)) != 4:
        raise ValueError(f"expected [N,4] coords, got {tuple(coords.shape)}")
    bev_coords = coords[:, [0, 2, 3]].contiguous()
    unique_coords, inverse = torch.unique(bev_coords, dim=0, return_inverse=True)
    unique_features = features.new_zeros((int(unique_coords.size(0)), int(features.size(1))))
    unique_features.index_add_(0, inverse, features)
    return unique_features, unique_coords, (int(spatial_shape_zyx[1]), int(spatial_shape_zyx[2]))


class _TorchSparseVoxelNeXtBevTail(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        import spconv.pytorch as spconv
        import torchsparse
        import torchsparse.nn as spnn

        self._torchsparse = torchsparse
        self._spconv = spconv
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.shared_conv = nn.Sequential(
            spnn.Conv3d(out_channels, out_channels, kernel_size=(3, 3, 1), bias=True),
            spnn.BatchNorm(out_channels),
            spnn.ReLU(),
        )

    def forward(
        self,
        features: torch.Tensor,
        coords_bzyx: torch.Tensor,
        *,
        spatial_shape_zyx: tuple[int, int, int],
        batch_size: int,
    ) -> SparseBEVOutput:
        bev_features, bev_coords, spatial_shape_hw = _collapse_to_bev_sparse(features, coords_bzyx, spatial_shape_zyx)
        bev_sp = self._spconv.SparseConvTensor(
            features=bev_features,
            indices=bev_coords,
            spatial_shape=list(spatial_shape_hw),
            batch_size=int(batch_size),
        )
        bev_sp = self.conv_out(bev_sp)
        ts_coords = torch.cat(
            (
                bev_sp.indices,
                torch.zeros((int(bev_sp.indices.size(0)), 1), device=bev_sp.indices.device, dtype=bev_sp.indices.dtype),
            ),
            dim=1,
        )
        ts = self._torchsparse.SparseTensor(
            feats=bev_sp.features,
            coords=ts_coords,
            spatial_range=(int(batch_size), int(spatial_shape_hw[0]), int(spatial_shape_hw[1]), 1),
        )
        ts = self.shared_conv(ts)
        return SparseBEVOutput(
            bev_features=ts.feats,
            bev_coords=ts.coords[:, :3].contiguous(),
            spatial_shape_hw=spatial_shape_hw,
            encoded_stride=8,
            batch_size=int(batch_size),
            backend="torchsparse",
        )


class _GTSparseVoxelNeXtBevTail(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_out = GeometricTemplateKernel9Conv3d(
            in_channels,
            out_channels,
            stride=1,
            padding=(1, 1, 0),
            bias=False,
        )
        self.conv_out_bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.shared_conv = GeometricTemplateKernel9Conv3d(
            out_channels,
            out_channels,
            stride=1,
            padding=(1, 1, 0),
            bias=True,
            subm=True,
        )
        self.shared_conv_bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(
        self,
        features: torch.Tensor,
        coords_bzyx: torch.Tensor,
        *,
        spatial_shape_zyx: tuple[int, int, int],
        batch_size: int,
    ) -> SparseBEVOutput:
        bev_features, bev_coords, spatial_shape_hw = _collapse_to_bev_sparse(
            features, coords_bzyx, spatial_shape_zyx
        )
        coords = torch.cat(
            (
                bev_coords,
                torch.zeros(
                    (bev_coords.size(0), 1),
                    device=bev_coords.device,
                    dtype=bev_coords.dtype,
                ),
            ),
            dim=1,
        )
        x = GTSparseSparseConvTensor(
            bev_features,
            coords,
            (*spatial_shape_hw, 1),
            int(batch_size),
        )
        x = self.conv_out(x)
        x.replace_feature_(self.relu(self.conv_out_bn(x.features)))
        x = self.shared_conv(x)
        x.replace_feature_(self.relu(self.shared_conv_bn(x.features)))
        return SparseBEVOutput(
            bev_features=x.features,
            bev_coords=x.indices[:, :3].contiguous(),
            spatial_shape_hw=spatial_shape_hw,
            encoded_stride=8,
            batch_size=int(batch_size),
            backend="gtsparse",
        )


class _SpconvVoxelNeXtBevTail(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self._spconv = spconv
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv2d(in_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )
        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            nn.ReLU(True),
        )

    def forward(
        self,
        features: torch.Tensor,
        coords_bzyx: torch.Tensor,
        *,
        spatial_shape_zyx: tuple[int, int, int],
        batch_size: int,
    ) -> SparseBEVOutput:
        bev_features, bev_coords, spatial_shape_hw = _collapse_to_bev_sparse(features, coords_bzyx, spatial_shape_zyx)
        bev = self._spconv.SparseConvTensor(
            features=bev_features,
            indices=bev_coords,
            spatial_shape=list(spatial_shape_hw),
            batch_size=int(batch_size),
        )
        bev = self.conv_out(bev)
        bev = self.shared_conv(bev)
        return SparseBEVOutput(
            bev_features=bev.features,
            bev_coords=bev.indices,
            spatial_shape_hw=spatial_shape_hw,
            encoded_stride=8,
            batch_size=int(batch_size),
            backend="spconv",
        )


def _upsample_to_coords(coords: torch.Tensor, *, factor: int) -> torch.Tensor:
    out = coords.clone()
    out[:, 1:] *= factor
    return out


def _contract_minkowski_coords(coords: torch.Tensor, *, base_stride) -> torch.Tensor:
    out = coords.clone()
    stride = torch.tensor(tuple(int(v) for v in base_stride), device=out.device, dtype=out.dtype)
    out[:, 1:] = torch.div(out[:, 1:], stride.view(1, 3), rounding_mode="floor")
    return out


def _stride2_shape(shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple((int(v) + 1) // 2 for v in shape_zyx)


def _replace_features_generic(x, new_features):
    if isinstance(x, GTSparseSparseConvTensor):
        x.replace_feature_(new_features)
        return x
    return x.replace_feature(new_features)


class _GTResidualBlock(nn.Module):
    def __init__(self, channels: int, *, sorted: bool = False) -> None:
        super().__init__()
        self.conv1 = GeometricTemplateSubMConv3d(channels, channels, 3, padding=1, bias=False, sorted=bool(sorted))
        self.bn1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv2 = GeometricTemplateSubMConv3d(channels, channels, 3, padding=1, bias=False, sorted=bool(sorted))
        self.bn2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        identity = x.features
        out = self.conv1(x)
        out.replace_feature_(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out.replace_feature_(self.relu(self.bn2(out.features) + identity))
        return out


class _GTConvBnReLU(nn.Module):
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
    def __init__(self, in_channels: int, out_channels: int, *, sorted: bool = False) -> None:
        super().__init__()
        self.conv = GeometricTemplateSparseConv3d(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            bias=False,
            build_reverse=False,
            sorted=bool(sorted),
        )
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.block1 = _GTResidualBlock(out_channels, sorted=bool(sorted))
        self.block2 = _GTResidualBlock(out_channels, sorted=bool(sorted))

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        x = self.conv(x)
        x.replace_feature_(self.relu(self.bn(x.features)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class GeometricTemplateVoxelNeXtBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: NuScenesVoxelNeXtModelConfig, *, sorted: bool = False) -> None:
        super().__init__()
        ch = model_cfg.channels
        self.model_name = "nuscenes_voxelnext_geometric_template"
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.sparse_shape_list = list(self.sparse_shape)
        self.conv_input = _GTConvBnReLU(
            GeometricTemplateSubMConv3d(
                model_cfg.input_channels,
                ch[0],
                3,
                padding=1,
                bias=False,
                sorted=bool(sorted),
            ),
            ch[0],
        )
        self.conv1 = nn.Sequential(_GTResidualBlock(ch[0], sorted=bool(sorted)), _GTResidualBlock(ch[0], sorted=bool(sorted)))
        self.conv2 = _GTDownStage(ch[0], ch[1], sorted=bool(sorted))
        self.conv3 = _GTDownStage(ch[1], ch[2], sorted=bool(sorted))
        self.conv4 = _GTDownStage(ch[2], ch[3], sorted=bool(sorted))
        self.conv5 = _GTDownStage(ch[3], ch[4], sorted=bool(sorted))
        self.conv6 = _GTDownStage(ch[4], ch[4], sorted=bool(sorted))
        self.bev_tail = _GTSparseVoxelNeXtBevTail(ch[3], model_cfg.out_channels)

    def _forward_trunk_stages(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = _make_gtsparse_sparse_tensor(voxel_features, voxel_coords, batch_size, self.sparse_shape_list)
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return x4, x5, x6, batch_size

    def forward_trunk_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x4, x5, x6, batch_size = self._forward_trunk_stages(voxel_features, voxel_coords, batch_size)
        return (
            torch.cat((x4.features, x5.features, x6.features), dim=0),
            torch.cat((x4.indices, _upsample_to_coords(x5.indices, factor=2), _upsample_to_coords(x6.indices, factor=4)), dim=0),
            tuple(x4.spatial_shape),
            batch_size,
        )

    def forward_trunk(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseTrunkOutput:
        features, coords_bzyx, spatial_shape_zyx, batch_size = self.forward_trunk_raw(voxel_features, voxel_coords, batch_size)
        return SparseTrunkOutput(
            features=features,
            coords_bzyx=coords_bzyx,
            spatial_shape_zyx=spatial_shape_zyx,
            encoded_stride=8,
            batch_size=batch_size,
            backend="gtsparse",
        )

    def forward_bev_tail(self, trunk: SparseTrunkOutput) -> SparseBEVOutput:
        out = self.bev_tail(
            trunk.features,
            trunk.coords_bzyx,
            spatial_shape_zyx=trunk.spatial_shape_zyx,
            batch_size=trunk.batch_size,
        )
        return SparseBEVOutput(
            bev_features=out.bev_features,
            bev_coords=out.bev_coords,
            spatial_shape_hw=out.spatial_shape_hw,
            encoded_stride=8,
            batch_size=trunk.batch_size,
            backend="gtsparse",
        )

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBEVOutput:
        return self.forward_bev_tail(self.forward_trunk(voxel_features, voxel_coords, batch_size))


class _TorchSparseResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        import torchsparse.nn as spnn

        self.conv1 = spnn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv2 = spnn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        identity = x.feats
        out = self.conv1(x)
        out.feats = self.relu(self.bn1(out.feats))
        out = self.conv2(out)
        out.feats = self.relu(self.bn2(out.feats) + identity)
        return out


class TorchSparseVoxelNeXtBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: NuScenesVoxelNeXtModelConfig) -> None:
        super().__init__()
        import torchsparse
        import torchsparse.nn as spnn

        self._torchsparse = torchsparse
        ch = model_cfg.channels
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.conv_input = nn.Sequential(spnn.Conv3d(model_cfg.input_channels, ch[0], 3, padding=1, bias=False), spnn.BatchNorm(ch[0]), spnn.ReLU())
        self.conv1 = nn.Sequential(_TorchSparseResidualBlock(ch[0]), _TorchSparseResidualBlock(ch[0]))
        self.conv2 = nn.Sequential(spnn.Conv3d(ch[0], ch[1], 3, stride=2, padding=1, bias=False), spnn.BatchNorm(ch[1]), spnn.ReLU(), _TorchSparseResidualBlock(ch[1]), _TorchSparseResidualBlock(ch[1]))
        self.conv3 = nn.Sequential(spnn.Conv3d(ch[1], ch[2], 3, stride=2, padding=1, bias=False), spnn.BatchNorm(ch[2]), spnn.ReLU(), _TorchSparseResidualBlock(ch[2]), _TorchSparseResidualBlock(ch[2]))
        self.conv4 = nn.Sequential(spnn.Conv3d(ch[2], ch[3], 3, stride=2, padding=1, bias=False), spnn.BatchNorm(ch[3]), spnn.ReLU(), _TorchSparseResidualBlock(ch[3]), _TorchSparseResidualBlock(ch[3]))
        self.conv5 = nn.Sequential(spnn.Conv3d(ch[3], ch[4], 3, stride=2, padding=1, bias=False), spnn.BatchNorm(ch[4]), spnn.ReLU(), _TorchSparseResidualBlock(ch[4]), _TorchSparseResidualBlock(ch[4]))
        self.conv6 = nn.Sequential(spnn.Conv3d(ch[4], ch[4], 3, stride=2, padding=1, bias=False), spnn.BatchNorm(ch[4]), spnn.ReLU(), _TorchSparseResidualBlock(ch[4]), _TorchSparseResidualBlock(ch[4]))
        self.bev_tail = _TorchSparseVoxelNeXtBevTail(ch[3], model_cfg.out_channels)

    def _forward_trunk_stages(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        spatial_range = (int(batch_size), *self.sparse_shape)
        x = self._torchsparse.SparseTensor(voxel_features, voxel_coords, spatial_range=spatial_range)
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return x4, x5, x6, batch_size

    def forward_trunk_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x4, x5, x6, batch_size = self._forward_trunk_stages(voxel_features, voxel_coords, batch_size)
        return (
            torch.cat((x4.feats, x5.feats, x6.feats), dim=0),
            torch.cat((x4.coords, _upsample_to_coords(x5.coords, factor=2), _upsample_to_coords(x6.coords, factor=4)), dim=0),
            tuple(int(v) for v in x4.spatial_range[1:]),
            int(batch_size),
        )

    def forward_trunk(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseTrunkOutput:
        features, coords_bzyx, spatial_shape_zyx, batch_size = self.forward_trunk_raw(voxel_features, voxel_coords, batch_size)
        return SparseTrunkOutput(
            features=features,
            coords_bzyx=coords_bzyx,
            spatial_shape_zyx=spatial_shape_zyx,
            encoded_stride=8,
            batch_size=batch_size,
            backend="torchsparse",
        )

    def forward_bev_tail(self, trunk: SparseTrunkOutput) -> SparseBEVOutput:
        return self.bev_tail(
            trunk.features,
            trunk.coords_bzyx,
            spatial_shape_zyx=trunk.spatial_shape_zyx,
            batch_size=int(trunk.batch_size),
        )

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBEVOutput:
        return self.forward_bev_tail(self.forward_trunk(voxel_features, voxel_coords, batch_size))


class _SpconvResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.conv1 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = _replace_features_generic(out, self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = _replace_features_generic(out, self.relu(self.bn2(out.features) + x.features))
        return out


class _SpconvDownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.conv = spconv.SparseConv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.block1 = _SpconvResidualBlock(out_channels)
        self.block2 = _SpconvResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = _replace_features_generic(x, self.relu(self.bn(x.features)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class SpconvVoxelNeXtBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: NuScenesVoxelNeXtModelConfig) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self._spconv = spconv
        ch = model_cfg.channels
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.conv_input = spconv.SparseSequential(spconv.SubMConv3d(model_cfg.input_channels, ch[0], 3, padding=1, bias=False), nn.BatchNorm1d(ch[0], eps=1e-3, momentum=0.01), nn.ReLU())
        self.conv1 = nn.Sequential(_SpconvResidualBlock(ch[0]), _SpconvResidualBlock(ch[0]))
        self.conv2 = _SpconvDownStage(ch[0], ch[1])
        self.conv3 = _SpconvDownStage(ch[1], ch[2])
        self.conv4 = _SpconvDownStage(ch[2], ch[3])
        self.conv5 = _SpconvDownStage(ch[3], ch[4])
        self.conv6 = _SpconvDownStage(ch[4], ch[4])
        self.bev_tail = _SpconvVoxelNeXtBevTail(ch[3], model_cfg.out_channels)

    def _forward_trunk_stages(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._spconv.SparseConvTensor(voxel_features, voxel_coords, list(self.sparse_shape), int(batch_size))
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return x4, x5, x6, batch_size

    def forward_trunk_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x4, x5, x6, batch_size = self._forward_trunk_stages(voxel_features, voxel_coords, batch_size)
        return (
            torch.cat((x4.features, x5.features, x6.features), dim=0),
            torch.cat((x4.indices, _upsample_to_coords(x5.indices, factor=2), _upsample_to_coords(x6.indices, factor=4)), dim=0),
            tuple(int(v) for v in x4.spatial_shape),
            int(batch_size),
        )

    def forward_trunk(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseTrunkOutput:
        features, coords_bzyx, spatial_shape_zyx, batch_size = self.forward_trunk_raw(voxel_features, voxel_coords, batch_size)
        return SparseTrunkOutput(
            features=features,
            coords_bzyx=coords_bzyx,
            spatial_shape_zyx=spatial_shape_zyx,
            encoded_stride=8,
            batch_size=batch_size,
            backend="spconv",
        )

    def forward_bev_tail(self, trunk: SparseTrunkOutput) -> SparseBEVOutput:
        return self.bev_tail(
            trunk.features,
            trunk.coords_bzyx,
            spatial_shape_zyx=trunk.spatial_shape_zyx,
            batch_size=int(trunk.batch_size),
        )

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBEVOutput:
        return self.forward_bev_tail(self.forward_trunk(voxel_features, voxel_coords, batch_size))


class _MinkowskiResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self.conv1 = ME.MinkowskiConvolution(channels, channels, kernel_size=3, stride=1, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(channels, eps=1e-3, momentum=0.01)
        self.relu = ME.MinkowskiReLU()
        self.conv2 = ME.MinkowskiConvolution(channels, channels, kernel_size=3, stride=1, dimension=3)
        self.bn2 = ME.MinkowskiBatchNorm(channels, eps=1e-3, momentum=0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class MinkowskiVoxelNeXtBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: NuScenesVoxelNeXtModelConfig) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self._ME = ME
        ch = model_cfg.channels
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.conv4_shape = _stride2_shape(_stride2_shape(_stride2_shape(tuple(int(v) for v in self.sparse_shape))))
        self.conv_input = nn.Sequential(ME.MinkowskiConvolution(model_cfg.input_channels, ch[0], kernel_size=3, stride=1, dimension=3), ME.MinkowskiBatchNorm(ch[0], eps=1e-3, momentum=0.01), ME.MinkowskiReLU())
        self.conv1 = nn.Sequential(_MinkowskiResidualBlock(ch[0]), _MinkowskiResidualBlock(ch[0]))
        self.conv2 = nn.Sequential(ME.MinkowskiConvolution(ch[0], ch[1], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[1], eps=1e-3, momentum=0.01), ME.MinkowskiReLU(), _MinkowskiResidualBlock(ch[1]), _MinkowskiResidualBlock(ch[1]))
        self.conv3 = nn.Sequential(ME.MinkowskiConvolution(ch[1], ch[2], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[2], eps=1e-3, momentum=0.01), ME.MinkowskiReLU(), _MinkowskiResidualBlock(ch[2]), _MinkowskiResidualBlock(ch[2]))
        self.conv4 = nn.Sequential(ME.MinkowskiConvolution(ch[2], ch[3], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[3], eps=1e-3, momentum=0.01), ME.MinkowskiReLU(), _MinkowskiResidualBlock(ch[3]), _MinkowskiResidualBlock(ch[3]))
        self.conv5 = nn.Sequential(ME.MinkowskiConvolution(ch[3], ch[4], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[4], eps=1e-3, momentum=0.01), ME.MinkowskiReLU(), _MinkowskiResidualBlock(ch[4]), _MinkowskiResidualBlock(ch[4]))
        self.conv6 = nn.Sequential(ME.MinkowskiConvolution(ch[4], ch[4], kernel_size=3, stride=2, dimension=3), ME.MinkowskiBatchNorm(ch[4], eps=1e-3, momentum=0.01), ME.MinkowskiReLU(), _MinkowskiResidualBlock(ch[4]), _MinkowskiResidualBlock(ch[4]))
        self.bev_tail = _TorchSparseVoxelNeXtBevTail(ch[3], model_cfg.out_channels)

    def _forward_trunk_stages(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._ME.SparseTensor(voxel_features, coordinates=voxel_coords)
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return x4, x5, x6, batch_size

    def forward_trunk_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x4, x5, x6, batch_size = self._forward_trunk_stages(voxel_features, voxel_coords, batch_size)
        base_stride = x4.tensor_stride
        return (
            torch.cat((x4.features, x5.features, x6.features), dim=0),
            torch.cat((
                _contract_minkowski_coords(x4.coordinates, base_stride=base_stride),
                _contract_minkowski_coords(x5.coordinates, base_stride=base_stride),
                _contract_minkowski_coords(x6.coordinates, base_stride=base_stride),
            ), dim=0),
            tuple(int(v) for v in self.conv4_shape),
            int(batch_size),
        )

    def forward_trunk(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseTrunkOutput:
        features, coords_bzyx, spatial_shape_zyx, batch_size = self.forward_trunk_raw(voxel_features, voxel_coords, batch_size)
        return SparseTrunkOutput(
            features=features,
            coords_bzyx=coords_bzyx,
            spatial_shape_zyx=spatial_shape_zyx,
            encoded_stride=8,
            batch_size=batch_size,
            backend="minkowski",
        )

    def forward_bev_tail(self, trunk: SparseTrunkOutput) -> SparseBEVOutput:
        out = self.bev_tail(
            trunk.features,
            trunk.coords_bzyx,
            spatial_shape_zyx=trunk.spatial_shape_zyx,
            batch_size=int(trunk.batch_size),
        )
        return SparseBEVOutput(
            bev_features=out.bev_features,
            bev_coords=out.bev_coords,
            spatial_shape_hw=out.spatial_shape_hw,
            encoded_stride=8,
            batch_size=int(trunk.batch_size),
            backend="minkowski",
        )

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBEVOutput:
        return self.forward_bev_tail(self.forward_trunk(voxel_features, voxel_coords, batch_size))


class VoxelNeXtSparseHead(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_channels, input_channels, bias=False), nn.BatchNorm1d(input_channels, eps=1e-3, momentum=0.01), nn.ReLU())
        self.hm = nn.Linear(input_channels, num_classes)
        self.center = nn.Linear(input_channels, 2)
        self.center_z = nn.Linear(input_channels, 1)
        self.dim = nn.Linear(input_channels, 3)
        self.rot = nn.Linear(input_channels, 2)
        nn.init.constant_(self.hm.bias, -2.19)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(features)
        return {
            "hm": self.hm(shared),
            "center": self.center(shared),
            "center_z": self.center_z(shared),
            "dim": self.dim(shared),
            "rot": self.rot(shared),
        }


def create_sparse_backbone(
    backend: str,
    *,
    data_cfg: KittiSecondDataConfig,
    model_cfg: NuScenesVoxelNeXtModelConfig,
    sorted: bool = False,
) -> nn.Module:
    name = str(backend).lower()
    if name == "gtsparse":
        return GeometricTemplateVoxelNeXtBackbone(data_cfg, model_cfg, sorted=bool(sorted))
    if name == "torchsparse":
        return TorchSparseVoxelNeXtBackbone(data_cfg, model_cfg)
    if name == "spconv":
        return SpconvVoxelNeXtBackbone(data_cfg, model_cfg)
    if name == "minkowski":
        return MinkowskiVoxelNeXtBackbone(data_cfg, model_cfg)
    raise KeyError(f"unsupported backend {backend!r}")


class NuScenesVoxelNeXtModel(nn.Module):
    def __init__(self, *, backend: str, config: NuScenesVoxelNeXtConfig | None = None, sorted: bool = False) -> None:
        super().__init__()
        self.config = config if config is not None else NuScenesVoxelNeXtConfig()
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
        self.pred_head = VoxelNeXtSparseHead(self.config.model.out_channels, len(self.config.data.class_names))

    def encode_batch(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        if isinstance(batch, KittiSecondBatch):
            voxels, voxel_coords, voxel_num_points, batch_size = batch.voxels, batch.voxel_coords, batch.voxel_num_points, int(batch.batch_size)
        else:
            voxels, voxel_coords, voxel_num_points, batch_size = batch["voxels"], batch["voxel_coords"], batch["voxel_num_points"], int(batch["batch_size"])
        return self.vfe(voxels, voxel_num_points), voxel_coords, batch_size

    def forward_sparse_backbone(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseTrunkOutput:
        return self.sparse_backbone.forward_trunk(voxel_features, voxel_coords, batch_size)

    def forward_sparse_backbone_raw(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self.sparse_backbone.forward_trunk_raw(voxel_features, voxel_coords, batch_size)

    def forward_sparse_convolutions(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> SparseBEVOutput:
        return self.sparse_backbone(voxel_features, voxel_coords, batch_size)

    def forward_sparse_backbone_from_batch(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> SparseTrunkOutput:
        voxel_features, voxel_coords, batch_size = self.encode_batch(batch)
        return self.forward_sparse_backbone(voxel_features, voxel_coords, batch_size)

    def forward_bev_tail_from_sparse_output(self, sparse_output: SparseTrunkOutput) -> SparseBEVOutput:
        return self.sparse_backbone.forward_bev_tail(sparse_output)

    def forward(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sparse_output = self.forward_sparse_backbone_from_batch(batch)
        bev_output = self.forward_bev_tail_from_sparse_output(sparse_output)
        predictions = self.pred_head(bev_output.bev_features)
        predictions["voxel_indices"] = bev_output.bev_coords
        predictions["spatial_shape"] = bev_output.spatial_shape_hw
        predictions["batch_size"] = bev_output.batch_size
        return predictions

    def decode_predictions(self, predictions: dict[str, torch.Tensor]) -> DecodedSecondPredictions:
        hm = torch.sigmoid(predictions["hm"])
        center = predictions["center"]
        center_z = predictions["center_z"]
        dim = torch.exp(torch.clamp(predictions["dim"], min=-5.0, max=5.0))
        rot = predictions["rot"]
        coords = predictions["voxel_indices"]
        batch_size = int(predictions["batch_size"])
        scores_out = hm.new_zeros((batch_size, 0))
        labels_out = coords.new_zeros((batch_size, 0))
        boxes_out = center.new_zeros((batch_size, 0, 7))
        batch_scores = []
        batch_labels = []
        batch_boxes = []
        stride = float(self.config.model.anchor_feature_map_stride)
        pc_range = self.config.data.point_cloud_range
        voxel_size = self.config.data.voxel_size
        for batch_idx in range(batch_size):
            mask = coords[:, 0] == batch_idx
            cur_hm = hm[mask]
            cur_center = center[mask]
            cur_center_z = center_z[mask]
            cur_dim = dim[mask]
            cur_rot = rot[mask]
            cur_coords = coords[mask]
            if int(cur_hm.numel()) == 0:
                batch_scores.append(hm.new_empty((0,)))
                batch_labels.append(coords.new_empty((0,), dtype=torch.long))
                batch_boxes.append(center.new_empty((0, 7)))
                continue
            flat_scores = cur_hm.view(-1)
            class_ids = torch.arange(cur_hm.size(1), device=cur_hm.device).view(1, -1).expand(cur_hm.size(0), -1).reshape(-1)
            point_ids = torch.arange(cur_hm.size(0), device=cur_hm.device).view(-1, 1).expand(-1, cur_hm.size(1)).reshape(-1)
            xs = (cur_coords[:, 2].float() + cur_center[:, 0]) * stride * float(voxel_size[0]) + float(pc_range[0])
            ys = (cur_coords[:, 1].float() + cur_center[:, 1]) * stride * float(voxel_size[1]) + float(pc_range[1])
            angle = torch.atan2(cur_rot[:, 1], cur_rot[:, 0])
            boxes = torch.cat((xs.unsqueeze(1), ys.unsqueeze(1), cur_center_z, cur_dim, angle.unsqueeze(1)), dim=1)
            batch_scores.append(flat_scores)
            batch_labels.append(class_ids + 1)
            batch_boxes.append(boxes.index_select(0, point_ids))
        max_len = max((int(v.numel()) for v in batch_scores), default=0)
        if max_len == 0:
            return DecodedSecondPredictions(boxes=boxes_out, scores=scores_out, labels=labels_out)
        scores_out = hm.new_zeros((batch_size, max_len))
        labels_out = coords.new_zeros((batch_size, max_len), dtype=torch.long)
        boxes_out = center.new_zeros((batch_size, max_len, 7))
        for batch_idx in range(batch_size):
            cur_scores = batch_scores[batch_idx]
            cur_labels = batch_labels[batch_idx]
            cur_boxes = batch_boxes[batch_idx]
            if int(cur_scores.numel()) == 0:
                continue
            scores_out[batch_idx, : cur_scores.numel()] = cur_scores
            labels_out[batch_idx, : cur_labels.numel()] = cur_labels
            boxes_out[batch_idx, : cur_boxes.size(0)] = cur_boxes
        return DecodedSecondPredictions(boxes=boxes_out, scores=scores_out, labels=labels_out)

    def postprocess_topk(self, predictions: dict[str, torch.Tensor], *, topk: int = 100) -> list[dict[str, torch.Tensor]]:
        return select_topk_predictions(self.decode_predictions(predictions), topk=topk)

    def postprocess_nms(self, predictions: dict[str, torch.Tensor], *, score_thresh: float = 0.1, nms_thresh: float = 0.01, pre_maxsize: int = 4096, post_maxsize: int = 500) -> list[dict[str, torch.Tensor]]:
        return select_nms_predictions(self.decode_predictions(predictions), score_thresh=score_thresh, nms_thresh=nms_thresh, pre_maxsize=pre_maxsize, post_maxsize=post_maxsize)


def _log_dir_for_run(*, root_log_dir: Path, device: str, model: NuScenesVoxelNeXtModel, data_root: Path, sweeps: int) -> Path:
    gpu_model = _sanitize_path_component(torch.cuda.get_device_name(torch.device(device).index or torch.cuda.current_device()))
    dtype_name = _sanitize_path_component(str(next(model.parameters()).dtype).replace("torch.", ""))
    dataset_name = _sanitize_path_component(data_root.name)
    return Path(root_log_dir) / f"logs_{gpu_model}_{dtype_name}_voxelnext_{dataset_name}_sweeps{int(sweeps)}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="nuScenes + VoxelNeXt thin runner")
    parser.add_argument("--backend", type=str, default="gtsparse")
    parser.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16"))
    parser.add_argument("--sorted", action="store_true", help="Enable sorted sparse-conv variants for GTSparse and TorchSparse backends.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/nuscenes"))
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
    parser.add_argument("--random-sample", action="store_true")
    parser.add_argument("--sweeps", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    runtime_dtype = resolve_runtime_dtype(args.dtype)
    require_cuda_device(args.device)
    if str(args.backend).lower() == "minkowski" and runtime_dtype == torch.float16:
        raise ValueError("Minkowski backend does not support fp16 in this runner")
    torchsparse_hash_rsv_ratio = None
    if str(args.backend).lower() == "torchsparse":
        torchsparse_hash_rsv_ratio = _configure_torchsparse_backend_for_nuscenes()
    config = NuScenesVoxelNeXtConfig(
        data=NuScenesVoxelNeXtDataConfig(
            root=discover_nuscenes_root(args.data_root),
            split=str(args.split),
            max_sweeps=max(1, int(getattr(args, "sweeps", NuScenesVoxelNeXtDataConfig().max_sweeps))),
        )
    )
    model = NuScenesVoxelNeXtModel(
        backend=args.backend,
        config=config,
        sorted=bool(getattr(args, "sorted", False)),
    ).to(args.device)
    if runtime_dtype == torch.float16:
        model = model.half()
    model.eval()
    dataset = _make_nuscenes_dataset(config.data)
    indices = _iter_sample_indices(
        dataset,
        frame_id=str(args.frame),
        num_samples=int(args.frames),
        random_sample=bool(getattr(args, "random_sample", False)),
    )
    loader = _make_loader(dataset, indices, config.data, batch_size=int(args.batch))
    warmup_batches = min(int(args.warmup), len(loader))
    log_dir = _log_dir_for_run(root_log_dir=Path(getattr(args, "log_dir", Path("logs"))), device=str(args.device), model=model, data_root=config.data.root, sweeps=int(config.data.max_sweeps))
    log_path = log_dir / f"{args.backend}.jsonl"
    config_path = log_dir / f"{args.backend}.config.json"
    summary_path = log_dir / f"{args.backend}.summary.json"
    device_index = torch.device(args.device).index
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device() if device_index is None else device_index)
    workload = f"voxelnext_nuscenes_sweeps{int(config.data.max_sweeps)}"
    min_template = int(os.environ.get("GTSPARSE_MIN_TEMPLATE", "0"))
    _write_json_file(
        config_path,
        {
            "backend": str(args.backend),
            "batch": int(args.batch),
            "data_root": str(config.data.root),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "sorted": bool(getattr(args, "sorted", False)),
            "frame": str(args.frame),
            "frames": int(args.frames),
            "gpu": gpu_name,
            "log_dir": str(log_dir),
            "min_template": min_template,
            "nms_thresh": float(args.nms_thresh),
            "post_maxsize": int(args.post_maxsize),
            "random_sample": bool(getattr(args, "random_sample", False)),
            "score_thresh": float(args.score_thresh),
            "spconv_do_sort": True,
            "split": str(args.split),
            "sweeps": int(config.data.max_sweeps),
            "timing_repeats": int(max(1, args.timing_repeats)),
            "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
            "topk": int(args.topk),
            "torchsparse_hash_rsv_ratio": torchsparse_hash_rsv_ratio,
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
            progress_desc=f"voxelnext/nuscenes/sw{int(config.data.max_sweeps)}/{args.backend}",
            on_result=lambda record: _append_backend_log_frame(log_file, record),
        )
    conv_only_times = [float(record["conv_only_ms"]) for record in results]
    end2end_times = [float(record["end2end_ms"]) for record in results]
    _write_json_file(
        summary_path,
        {
            "backend": str(args.backend),
            "batch": int(args.batch),
            "data_root": str(config.data.root),
            "dtype": str(args.dtype),
            "frames_logged": int(len(results)),
            "gpu": gpu_name,
            "min_template": min_template,
            "random_sample": bool(getattr(args, "random_sample", False)),
            "split": str(args.split),
            "spconv_do_sort": True,
            "sweeps": int(config.data.max_sweeps),
            "stats": {"conv_only": _stats_dict(conv_only_times), "end2end": _stats_dict(end2end_times)},
            "timing_repeats": int(max(1, args.timing_repeats)),
            "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
            "torchsparse_hash_rsv_ratio": torchsparse_hash_rsv_ratio,
            "warmup_batches": int(warmup_batches),
            "workload": workload,
        },
    )
    summary = {
        "backend": str(args.backend),
        "dtype": str(args.dtype),
        "device": str(args.device),
        "gpu": gpu_name,
        "data_root": str(config.data.root),
        "split": str(args.split),
        "sweeps": int(config.data.max_sweeps),
        "frames": int(len(indices)),
        "batch": int(args.batch),
        "timing_repeats": int(max(1, args.timing_repeats)),
        "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
        "torchsparse_hash_rsv_ratio": torchsparse_hash_rsv_ratio,
        "log_dir": str(log_dir),
        "min_template": min_template,
        "log_jsonl": str(log_path),
        "config_json": str(config_path),
        "summary_json": str(summary_path),
        "workload": workload,
        "results": results,
    }
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
    print(f"frames={summary['frames']} batch={summary['batch']}")
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
