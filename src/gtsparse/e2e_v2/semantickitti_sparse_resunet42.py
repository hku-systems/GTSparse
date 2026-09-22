"""SemanticKITTI + MinkUNet thin runner."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass, field
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as torch_data
from tqdm.auto import tqdm

import torchsparse

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torchsparse.backends.allow_tf32 = False

from gtsparse.e2e_v2.common import clear_sparse_metadata, measure_cuda_elapsed_ms, require_cuda_device, resolve_runtime_dtype
from gtsparse.sparse3d.geometric_template import (
    GeometricTemplateKernel8Conv3d,
    GeometricTemplateKernel8InverseConv3d,
    GeometricTemplateSparseConv3d,
    GeometricTemplateSparseInverseConv3d,
    GeometricTemplateSubMConv3d,
)
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor

from .kitti_second import (
    KittiLidarDataset,
    KittiSecondBatch,
    KittiSecondDataConfig,
    MeanVoxelFeatureEncoder,
    _configure_torchsparse_conv_sort_mode,
    _format_stats,
    _iter_sample_indices,
    _make_loader,
    _sanitize_path_component,
    _stats_dict,
    _write_json_file,
    discover_kitti_root,
)


def _replace_features_generic(x, new_features):
    if isinstance(x, GTSparseSparseConvTensor):
        x.replace_feature_(new_features)
        return x
    return x.replace_feature(new_features)


def _configure_spconv_do_sort(enabled: bool) -> None:
    try:
        constants = importlib.import_module("spconv.constants")
    except Exception:
        return
    constants.SPCONV_DO_SORT = bool(enabled)
    try:
        ops = importlib.import_module("spconv.pytorch.ops")
    except Exception:
        return
    ops.SPCONV_DO_SORT = bool(enabled)
    orig_name = "_ACTIMM_ORIG_GET_INDICE_PAIRS_IGEMM"
    orig = getattr(ops, orig_name, None)
    if orig is None:
        orig = ops.get_indice_pairs_implicit_gemm
        setattr(ops, orig_name, orig)

    def wrapped(
        indices,
        batch_size,
        spatial_shape,
        algo,
        ksize,
        stride,
        padding,
        dilation,
        out_padding,
        subm=False,
        transpose=False,
        is_train=True,
        alloc=None,
        timer=None,
        num_out_act_bound=-1,
        direct_table=None,
        do_sort=None,
    ):
        effective_direct_table = ops.SPCONV_USE_DIRECT_TABLE if direct_table is None else bool(direct_table)
        effective_do_sort = bool(enabled) if do_sort is None else bool(do_sort)
        if timer is None:
            timer = ops.CUDAKernelTimer(False)
        return orig(
            indices,
            batch_size,
            spatial_shape,
            algo,
            ksize,
            stride,
            padding,
            dilation,
            out_padding,
            subm=subm,
            transpose=transpose,
            is_train=is_train,
            alloc=alloc,
            timer=timer,
            num_out_act_bound=num_out_act_bound,
            direct_table=effective_direct_table,
            do_sort=effective_do_sort,
        )

    ops.get_indice_pairs_implicit_gemm = wrapped


@dataclass(frozen=True)
class SemanticKITTISparseResUNet42ModelConfig:
    input_channels: int = 4
    num_classes: int = 19
    stem_channels: int = 32
    encoder_channels: tuple[int, int, int, int] = (32, 64, 128, 256)
    decoder_channels: tuple[int, int, int, int] = (256, 128, 96, 96)


@dataclass(frozen=True)
class SemanticKITTISparseResUNet42Config:
    data: KittiSecondDataConfig = field(default_factory=lambda: SemanticKITTISparseResUNet42DataConfig())
    model: SemanticKITTISparseResUNet42ModelConfig = field(default_factory=SemanticKITTISparseResUNet42ModelConfig)


@dataclass(frozen=True)
class SemanticKITTISparseResUNet42DataConfig(KittiSecondDataConfig):
    root: Path = Path("dataset/semantickitti")
    split: str = "val"
    class_names: tuple[str, ...] = (
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
    )
    point_cloud_range: tuple[float, float, float, float, float, float] = (-75.2, -75.2, -4.0, 75.2, 75.2, 2.0)
    voxel_size: tuple[float, float, float] = (0.1, 0.1, 0.1)
    num_point_features: int = 4
    max_points_per_voxel: int = 10
    max_voxels_train: int = 160000
    max_voxels_eval: int = 200000
    max_sweeps: int = 1


def available_semantickitti_sparse_resunet42_backends() -> dict[str, dict[str, str | bool | None]]:
    from .kitti_second import available_kitti_second_backends

    return available_kitti_second_backends()


def discover_semantickitti_root(root: str | Path | None = None) -> Path:
    if root is None:
        return Path("dataset/semantickitti")
    return Path(root)


def _resolve_semantickitti_sequences_dir(root: str | Path) -> Path:
    base = discover_semantickitti_root(root)
    if (base / "dataset" / "sequences").exists():
        return base / "dataset" / "sequences"
    if (base / "sequences").exists():
        return base / "sequences"
    if base.name == "sequences" and base.exists():
        return base
    raise FileNotFoundError(f"Could not find SemanticKITTI sequences directory under {base}")


def _semantickitti_split_sequences(split: str) -> list[str]:
    split_name = str(split).lower()
    mapping = {
        "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        "val": ["08"],
        "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        "trainval": ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        "all": [f"{idx:02d}" for idx in range(22)],
    }
    try:
        return mapping[split_name]
    except KeyError as exc:
        raise KeyError(f"unsupported SemanticKITTI split {split!r}") from exc


def _read_semantickitti_points(path: Path) -> np.ndarray:
    points = np.fromfile(str(path), dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"SemanticKITTI lidar file {path} does not contain a multiple of 4 float32 values")
    return np.ascontiguousarray(points.reshape(-1, 4), dtype=np.float32)


def _parse_semantickitti_calibration(path: Path) -> dict[str, np.ndarray]:
    calib: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, content = line.split(":", 1)
            values = [float(v) for v in content.strip().split()]
            if len(values) != 12:
                continue
            pose = np.zeros((4, 4), dtype=np.float64)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            calib[key] = pose
    if "Tr" not in calib:
        raise KeyError(f"SemanticKITTI calibration file {path} does not contain 'Tr'")
    return calib


def _parse_semantickitti_poses(path: Path, calibration: dict[str, np.ndarray]) -> list[np.ndarray]:
    poses: list[np.ndarray] = []
    tr = calibration["Tr"]
    tr_inv = np.linalg.inv(tr)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            values = [float(v) for v in line.strip().split()]
            if len(values) != 12:
                continue
            pose = np.zeros((4, 4), dtype=np.float64)
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0
            poses.append((tr_inv @ pose @ tr).astype(np.float32))
    return poses


def _fuse_semantickitti_multiscan(points_xyzi: np.ndarray, pose_ref: np.ndarray, pose_src: np.ndarray) -> np.ndarray:
    hpoints = np.hstack((points_xyzi[:, :3], np.ones((int(points_xyzi.shape[0]), 1), dtype=np.float32)))
    transformed = (hpoints @ pose_src.T)[:, :3]
    transformed = transformed - pose_ref[:3, 3]
    aligned = transformed @ pose_ref[:3, :3]
    return np.ascontiguousarray(np.hstack((aligned, points_xyzi[:, 3:4])), dtype=np.float32)


class SemanticKITTILidarDataset(torch_data.Dataset):
    def __init__(self, root: str | Path, *, split: str = "val", max_sweeps: int = 1) -> None:
        super().__init__()
        self.root = discover_semantickitti_root(root)
        self.split = str(split)
        self.max_sweeps = max(1, int(max_sweeps))
        self.sequences_dir = _resolve_semantickitti_sequences_dir(self.root)
        self.sequence_ids = _semantickitti_split_sequences(self.split)
        self.calibrations: dict[str, dict[str, np.ndarray]] = {}
        self.poses: dict[str, list[np.ndarray]] = {}
        self.samples: list[tuple[str, str, int, Path]] = []
        self._scan_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._scan_cache_limit = max(32, int(self.max_sweeps) * 8)
        for seq in self.sequence_ids:
            calib_path = self.sequences_dir / seq / "calib.txt"
            poses_path = self.sequences_dir / seq / "poses.txt"
            if self.max_sweeps > 1:
                if not calib_path.exists() or not poses_path.exists():
                    raise FileNotFoundError(f"SemanticKITTI sequence {seq} needs calib.txt and poses.txt for sweeps={self.max_sweeps}")
                self.calibrations[seq] = _parse_semantickitti_calibration(calib_path)
                self.poses[seq] = _parse_semantickitti_poses(poses_path, self.calibrations[seq])
            velodyne_dir = self.sequences_dir / seq / "velodyne"
            if not velodyne_dir.exists():
                continue
            for path in sorted(velodyne_dir.glob("*.bin")):
                frame_id = f"{seq}_{path.stem}"
                self.samples.append((seq, frame_id, int(path.stem), path))
        self.sample_ids = [frame_id for _seq, frame_id, _frame_idx, _path in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_points_cached(self, path: Path) -> np.ndarray:
        key = str(path)
        cached = self._scan_cache.get(key)
        if cached is not None:
            self._scan_cache.move_to_end(key)
            return cached
        points = _read_semantickitti_points(path)
        self._scan_cache[key] = points
        if len(self._scan_cache) > self._scan_cache_limit:
            self._scan_cache.popitem(last=False)
        return points

    def _load_points(self, seq: str, frame_idx: int, path: Path) -> np.ndarray:
        points = [self._load_raw_points_cached(path)]
        if self.max_sweeps <= 1:
            return points[0]
        ref_pose = self.poses[seq][frame_idx]
        for offset in range(1, self.max_sweeps):
            src_idx = frame_idx - offset
            if src_idx < 0:
                break
            src_path = path.parent / f"{src_idx:06d}.bin"
            if not src_path.exists():
                break
            src_points = self._load_raw_points_cached(src_path)
            src_pose = self.poses[seq][src_idx]
            points.append(_fuse_semantickitti_multiscan(src_points, ref_pose, src_pose))
        return np.ascontiguousarray(np.concatenate(points, axis=0), dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq, frame_id, frame_idx, path = self.samples[index]
        return {
            "frame_id": frame_id,
            "sequence": seq,
            "points": self._load_points(seq, frame_idx, path),
        }


def _make_semantickitti_dataset(data_cfg: KittiSecondDataConfig):
    root = Path(data_cfg.root)
    if (root / "dataset" / "sequences").exists() or (root / "sequences").exists() or root.name == "sequences":
        return SemanticKITTILidarDataset(root, split=str(data_cfg.split), max_sweeps=int(getattr(data_cfg, "max_sweeps", 1)))
    return KittiLidarDataset(root, split=str(data_cfg.split))


def _make_gtsparse_sparse_tensor(voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int, sparse_shape_zyx) -> GTSparseSparseConvTensor:
    return GTSparseSparseConvTensor(voxel_features, voxel_coords, sparse_shape_zyx, batch_size)


def _cat_gtsparse(a: GTSparseSparseConvTensor, b: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
    return a.replace_feature_(torch.cat((a.features, b.features), dim=1))


_TORCHSPARSE_VIEW_KEY = "torchsparse_view"


class _GTResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, sorted: bool = False) -> None:
        super().__init__()
        self.conv1 = GeometricTemplateSubMConv3d(in_channels, out_channels, 3, padding=1, bias=False, sorted=bool(sorted))
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = GeometricTemplateSubMConv3d(out_channels, out_channels, 3, padding=1, bias=False, sorted=bool(sorted))
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else None

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        identity = x.features if self.shortcut is None else self.shortcut(x.features)
        out = self.conv1(x)
        out.replace_feature_(self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out.replace_feature_(self.relu(self.bn2(out.features) + identity))
        return out


def _make_gt_res_stack(*, in_channels: int, out_channels: int, blocks: int, sorted: bool = False) -> nn.Sequential:
    layers = [_GTResBlock(in_channels, out_channels, sorted=bool(sorted))]
    for _ in range(max(0, int(blocks) - 1)):
        layers.append(_GTResBlock(out_channels, out_channels, sorted=bool(sorted)))
    return nn.Sequential(*layers)


def _gtsparse_to_torchsparse(x: GTSparseSparseConvTensor):
    import torchsparse

    ts = x._runtime_cache.get(_TORCHSPARSE_VIEW_KEY)
    if ts is None:
        ts = torchsparse.SparseTensor(
            feats=x.features,
            coords=x.indices,
            spatial_range=(x.batch_size, *x.spatial_shape),
        )
        ts._caches.cmaps[ts.stride] = (ts.coords, ts.spatial_range)
        x._runtime_cache[_TORCHSPARSE_VIEW_KEY] = ts
        return ts
    ts.feats = x.features
    if ts.spatial_range is None:
        ts.spatial_range = (x.batch_size, *x.spatial_shape)
    ts._caches.cmaps.setdefault(ts.stride, (ts.coords, ts.spatial_range))
    return ts


def _torchsparse_spatial_range(x) -> tuple[int, int, int, int]:
    if x.spatial_range is not None:
        return tuple(x.spatial_range)
    cached = x._caches.cmaps.get(x.stride)
    if cached is not None and cached[1] is not None:
        return tuple(cached[1])
    raise RuntimeError("hybrid MinkUNet expected TorchSparse view to carry spatial_range/cmaps; reconstructing it from coords is disabled on the hot path")


def _torchsparse_to_gtsparse(x) -> GTSparseSparseConvTensor:
    spatial_range = _torchsparse_spatial_range(x)
    if x.coords.dtype != torch.int32:
        raise RuntimeError(f"hybrid MinkUNet expected TorchSparse coords to stay int32, got {x.coords.dtype}")
    return GTSparseSparseConvTensor._from_components(
        features=x.feats,
        coords=x.coords,
        spatial_shape=list(spatial_range[1:]),
        batch_size=spatial_range[0],
        indice_dict=None,
        runtime_cache={_TORCHSPARSE_VIEW_KEY: x},
        index_grid=None,
        coord_hashmap=None,
        metadata=None,
    )


class _GTMinkUNetDownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, blocks: int, sorted: bool = False) -> None:
        super().__init__()
        self.down = GeometricTemplateKernel8Conv3d(in_channels, in_channels)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(True)
        self.blocks = _make_gt_res_stack(
            in_channels=in_channels,
            out_channels=out_channels,
            blocks=blocks,
            sorted=bool(sorted),
        )

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        x = self.down(x)
        x.replace_feature_(self.relu(self.bn(x.features)))
        return self.blocks(x)


class _GTMinkUNetUpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, *, blocks: int, sorted: bool = False) -> None:
        super().__init__()
        self.up = GeometricTemplateKernel8InverseConv3d(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(True)
        self.blocks = _make_gt_res_stack(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
            blocks=blocks,
            sorted=bool(sorted),
        )

    def forward(self, x: GTSparseSparseConvTensor, skip: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        x = self.up(x)
        x.replace_feature_(self.relu(self.bn(x.features)))
        x = _cat_gtsparse(x, skip)
        return self.blocks(x)


class _GTDownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, sorted: bool = False) -> None:
        super().__init__()
        self.down = GeometricTemplateSparseConv3d(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            bias=False,
            sorted=bool(sorted),
        )
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.block1 = _GTResBlock(out_channels, out_channels, sorted=bool(sorted))
        self.block2 = _GTResBlock(out_channels, out_channels, sorted=bool(sorted))

    def forward(self, x: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        x = self.down(x)
        x.replace_feature_(self.relu(self.bn(x.features)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class _GTUpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, *, sorted: bool = False) -> None:
        super().__init__()
        self.up = GeometricTemplateSparseInverseConv3d(in_channels, out_channels, 3, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.fuse = GeometricTemplateSubMConv3d(
            out_channels + skip_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            sorted=bool(sorted),
        )
        self.fuse_bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.block1 = _GTResBlock(out_channels, out_channels, sorted=bool(sorted))
        self.block2 = _GTResBlock(out_channels, out_channels, sorted=bool(sorted))

    def forward(self, x: GTSparseSparseConvTensor, skip: GTSparseSparseConvTensor) -> GTSparseSparseConvTensor:
        x = self.up(x)
        x.replace_feature_(self.relu(self.bn(x.features)))
        x = _cat_gtsparse(x, skip)
        x = self.fuse(x)
        x.replace_feature_(self.relu(self.fuse_bn(x.features)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class GeometricTemplateSparseResUNet42Backbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig, *, sorted: bool = False) -> None:
        super().__init__()
        self.model_name = "semantickitti_sparse_resunet42_geometric_template"
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.sparse_shape_list = list(self.sparse_shape)
        stem = int(model_cfg.stem_channels)
        enc = tuple(int(v) for v in model_cfg.encoder_channels)
        dec = tuple(int(v) for v in model_cfg.decoder_channels)
        self.stem0 = GeometricTemplateSubMConv3d(model_cfg.input_channels, stem, 3, padding=1, bias=False, sorted=bool(sorted))
        self.stem0_bn = nn.BatchNorm1d(stem, eps=1e-3, momentum=0.01)
        self.stem1 = GeometricTemplateSubMConv3d(stem, stem, 3, padding=1, bias=False, sorted=bool(sorted))
        self.stem1_bn = nn.BatchNorm1d(stem, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.down1 = _GTDownStage(stem, enc[0], sorted=bool(sorted))
        self.down2 = _GTDownStage(enc[0], enc[1], sorted=bool(sorted))
        self.down3 = _GTDownStage(enc[1], enc[2], sorted=bool(sorted))
        self.down4 = _GTDownStage(enc[2], enc[3], sorted=bool(sorted))
        self.up1 = _GTUpStage(enc[3], dec[0], enc[2], sorted=bool(sorted))
        self.up2 = _GTUpStage(dec[0], dec[1], enc[1], sorted=bool(sorted))
        self.up3 = _GTUpStage(dec[1], dec[2], enc[0], sorted=bool(sorted))
        self.up4 = _GTUpStage(dec[2], dec[3], stem, sorted=bool(sorted))
        self.out_channels = dec[3]

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> GTSparseSparseConvTensor:
        x = _make_gtsparse_sparse_tensor(voxel_features, voxel_coords, batch_size, self.sparse_shape_list)
        x = self.stem0(x)
        x.replace_feature_(self.relu(self.stem0_bn(x.features)))
        x = self.stem1(x)
        x.replace_feature_(self.relu(self.stem1_bn(x.features)))
        s0 = x
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return x


class _TorchSparseResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        import torchsparse.nn as spnn

        self.conv1 = spnn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = spnn.BatchNorm(out_channels)
        self.relu = spnn.ReLU(True)
        self.conv2 = spnn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = spnn.BatchNorm(out_channels)
        self.shortcut = nn.Sequential(spnn.Conv3d(in_channels, out_channels, 1), spnn.BatchNorm(out_channels)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class TorchSparseSparseResUNet42Backbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig) -> None:
        super().__init__()
        import torchsparse
        from torchsparse.backbones import SparseResUNet42

        self._torchsparse = torchsparse
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.backbone = SparseResUNet42(in_channels=int(model_cfg.input_channels))
        self.out_channels = int(model_cfg.decoder_channels[-1])

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._torchsparse.SparseTensor(voxel_features, voxel_coords, spatial_range=(int(batch_size), *self.sparse_shape))
        outputs = self.backbone(x)
        return outputs[-1]


class _SpconvResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, indice_key: str | None = None) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.conv1 = spconv.SubMConv3d(in_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(out_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                out_channels,
                1,
                bias=False,
                indice_key=None if indice_key is None else indice_key + "_shortcut",
            ),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels else None

    def forward(self, x):
        identity = x.features if self.shortcut is None else self.shortcut(x).features
        out = self.conv1(x)
        out = _replace_features_generic(out, self.relu(self.bn1(out.features)))
        out = self.conv2(out)
        out = _replace_features_generic(out, self.relu(self.bn2(out.features) + identity))
        return out


class _SpconvDownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, indice_key: str) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.down = spconv.SparseConv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False, indice_key=indice_key)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.block1 = _SpconvResBlock(out_channels, out_channels, indice_key=indice_key + "_subm")
        self.block2 = _SpconvResBlock(out_channels, out_channels, indice_key=indice_key + "_subm")

    def forward(self, x):
        x = self.down(x)
        x = _replace_features_generic(x, self.relu(self.bn(x.features)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class _SpconvUpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int, *, indice_key: str) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self.up = spconv.SparseInverseConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()
        self.fuse = spconv.SubMConv3d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False, indice_key=indice_key + "_fuse")
        self.fuse_bn = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.block1 = _SpconvResBlock(out_channels, out_channels, indice_key=indice_key + "_fuse")
        self.block2 = _SpconvResBlock(out_channels, out_channels, indice_key=indice_key + "_fuse")

    def forward(self, x, skip):
        x = self.up(x)
        x = _replace_features_generic(x, self.relu(self.bn(x.features)))
        if int(x.features.size(0)) != int(skip.features.size(0)) or not torch.equal(x.indices, skip.indices):
            raise RuntimeError("spconv decoder expected inverse-conv output to match skip coordinates")
        fused = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
        x = self.fuse(fused)
        x = _replace_features_generic(x, self.relu(self.fuse_bn(x.features)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class SpconvSparseResUNet42Backbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        self._spconv = spconv
        self.sparse_shape = data_cfg.sparse_shape_zyx
        stem = int(model_cfg.stem_channels)
        enc = tuple(int(v) for v in model_cfg.encoder_channels)
        dec = tuple(int(v) for v in model_cfg.decoder_channels)
        self.stem = spconv.SparseSequential(spconv.SubMConv3d(model_cfg.input_channels, stem, 3, padding=1, bias=False, indice_key="stem"), nn.BatchNorm1d(stem, eps=1e-3, momentum=0.01), nn.ReLU(), spconv.SubMConv3d(stem, stem, 3, padding=1, bias=False, indice_key="stem"), nn.BatchNorm1d(stem, eps=1e-3, momentum=0.01), nn.ReLU())
        self.down1 = _SpconvDownStage(stem, enc[0], indice_key="down1")
        self.down2 = _SpconvDownStage(enc[0], enc[1], indice_key="down2")
        self.down3 = _SpconvDownStage(enc[1], enc[2], indice_key="down3")
        self.down4 = _SpconvDownStage(enc[2], enc[3], indice_key="down4")
        self.up1 = _SpconvUpStage(enc[3], dec[0], enc[2], indice_key="down4")
        self.up2 = _SpconvUpStage(dec[0], dec[1], enc[1], indice_key="down3")
        self.up3 = _SpconvUpStage(dec[1], dec[2], enc[0], indice_key="down2")
        self.up4 = _SpconvUpStage(dec[2], dec[3], stem, indice_key="down1")
        self.out_channels = dec[3]

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._spconv.SparseConvTensor(voxel_features, voxel_coords, list(self.sparse_shape), int(batch_size))
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return x


class _MinkowskiResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self.conv1 = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=1, dimension=3)
        self.bn1 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU()
        self.conv2 = ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, stride=1, dimension=3)
        self.bn2 = ME.MinkowskiBatchNorm(out_channels)
        self.shortcut = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, bias=False, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class _MinkowskiDownStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self.down = ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=2, dimension=3)
        self.bn = ME.MinkowskiBatchNorm(out_channels, eps=1e-3, momentum=0.01)
        self.relu = ME.MinkowskiReLU()
        self.block1 = _MinkowskiResBlock(out_channels, out_channels)
        self.block2 = _MinkowskiResBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.relu(self.bn(self.down(x)))
        x = self.block1(x)
        x = self.block2(x)
        return x


class _MinkowskiUpStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, skip_channels: int) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self._ME = ME
        self.up = ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=3, stride=2, dimension=3)
        self.bn = ME.MinkowskiBatchNorm(out_channels, eps=1e-3, momentum=0.01)
        self.relu = ME.MinkowskiReLU()
        self.fuse = _MinkowskiResBlock(out_channels + skip_channels, out_channels)
        self.block = _MinkowskiResBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.relu(self.bn(self.up(x)))
        x = self._ME.cat(x, skip)
        x = self.fuse(x)
        x = self.block(x)
        return x


class MinkowskiSparseResUNet42Backbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        self._ME = ME
        stem = int(model_cfg.stem_channels)
        enc = tuple(int(v) for v in model_cfg.encoder_channels)
        dec = tuple(int(v) for v in model_cfg.decoder_channels)
        self.stem = nn.Sequential(ME.MinkowskiConvolution(model_cfg.input_channels, stem, kernel_size=3, stride=1, dimension=3), ME.MinkowskiBatchNorm(stem, eps=1e-3, momentum=0.01), ME.MinkowskiReLU(), ME.MinkowskiConvolution(stem, stem, kernel_size=3, stride=1, dimension=3), ME.MinkowskiBatchNorm(stem, eps=1e-3, momentum=0.01), ME.MinkowskiReLU())
        self.down1 = _MinkowskiDownStage(stem, enc[0])
        self.down2 = _MinkowskiDownStage(enc[0], enc[1])
        self.down3 = _MinkowskiDownStage(enc[1], enc[2])
        self.down4 = _MinkowskiDownStage(enc[2], enc[3])
        self.up1 = _MinkowskiUpStage(enc[3], dec[0], enc[2])
        self.up2 = _MinkowskiUpStage(dec[0], dec[1], enc[1])
        self.up3 = _MinkowskiUpStage(dec[1], dec[2], enc[0])
        self.up4 = _MinkowskiUpStage(dec[2], dec[3], stem)
        self.out_channels = dec[3]

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        del batch_size
        x = self._ME.SparseTensor(voxel_features, coordinates=voxel_coords)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)
        return x


class GeometricTemplateMinkUNetBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig, *, sorted: bool = False) -> None:
        super().__init__()
        self.model_name = "semantickitti_minkunet_geometric_template"
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.sparse_shape_list = list(self.sparse_shape)
        p = MINKUNET_PLANES
        l = MINKUNET_LAYERS
        self.stem0 = GeometricTemplateSubMConv3d(model_cfg.input_channels, p[0], 3, padding=1, bias=False, sorted=bool(sorted))
        self.stem0_bn = nn.BatchNorm1d(p[0])
        self.stem1 = GeometricTemplateSubMConv3d(p[0], p[0], 3, padding=1, bias=False, sorted=bool(sorted))
        self.stem1_bn = nn.BatchNorm1d(p[0])
        self.relu = nn.ReLU()
        self.down1 = _GTMinkUNetDownStage(p[0], p[0], blocks=l[0], sorted=bool(sorted))
        self.down2 = _GTMinkUNetDownStage(p[0], p[1], blocks=l[1], sorted=bool(sorted))
        self.down3 = _GTMinkUNetDownStage(p[1], p[2], blocks=l[2], sorted=bool(sorted))
        self.down4 = _GTMinkUNetDownStage(p[2], p[3], blocks=l[3], sorted=bool(sorted))
        self.up4 = _GTMinkUNetUpStage(p[3], p[4], p[2], blocks=l[4], sorted=bool(sorted))
        self.up3 = _GTMinkUNetUpStage(p[4], p[5], p[1], blocks=l[5], sorted=bool(sorted))
        self.up2 = _GTMinkUNetUpStage(p[5], p[6], p[0], blocks=l[6], sorted=bool(sorted))
        self.up1 = _GTMinkUNetUpStage(p[6], p[7], p[0], blocks=l[7], sorted=bool(sorted))
        self.out_channels = p[7]

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int) -> GTSparseSparseConvTensor:
        x = _make_gtsparse_sparse_tensor(voxel_features, voxel_coords, batch_size, self.sparse_shape_list)
        x = self.stem0(x)
        x.replace_feature_(self.relu(self.stem0_bn(x.features)))
        out_p1 = self.stem1(x)
        out_p1.replace_feature_(self.relu(self.stem1_bn(out_p1.features)))
        out_b1p2 = self.down1(out_p1)
        out_b2p4 = self.down2(out_b1p2)
        out_b3p8 = self.down3(out_b2p4)
        out = self.down4(out_b3p8)
        out = self.up4(out, out_b3p8)
        out = self.up3(out, out_b2p4)
        out = self.up2(out, out_b1p2)
        out = self.up1(out, out_p1)
        return out


MINKUNET_PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
MINKUNET_LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


def _make_torchsparse_res_stack(*, in_channels: int, out_channels: int, blocks: int):
    from torchsparse.backbones.modules import SparseResBlock

    layers = [SparseResBlock(in_channels, out_channels, 3)]
    for _ in range(max(0, int(blocks) - 1)):
        layers.append(SparseResBlock(out_channels, out_channels, 3))
    return nn.Sequential(*layers)


class TorchSparseMinkUNetBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig) -> None:
        super().__init__()
        import torchsparse
        import torchsparse.nn as spnn

        p = MINKUNET_PLANES
        l = MINKUNET_LAYERS
        self._torchsparse = torchsparse
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.stem = nn.Sequential(
            spnn.Conv3d(model_cfg.input_channels, p[0], kernel_size=3),
            spnn.BatchNorm(p[0]),
            spnn.ReLU(True),
            spnn.Conv3d(p[0], p[0], kernel_size=3),
            spnn.BatchNorm(p[0]),
            spnn.ReLU(True),
        )
        self.down1 = nn.Sequential(spnn.Conv3d(p[0], p[0], kernel_size=2, stride=2, generative=False), spnn.BatchNorm(p[0]), spnn.ReLU(True))
        self.block1 = _make_torchsparse_res_stack(in_channels=p[0], out_channels=p[0], blocks=l[0])
        self.down2 = nn.Sequential(spnn.Conv3d(p[0], p[0], kernel_size=2, stride=2, generative=False), spnn.BatchNorm(p[0]), spnn.ReLU(True))
        self.block2 = _make_torchsparse_res_stack(in_channels=p[0], out_channels=p[1], blocks=l[1])
        self.down3 = nn.Sequential(spnn.Conv3d(p[1], p[1], kernel_size=2, stride=2, generative=False), spnn.BatchNorm(p[1]), spnn.ReLU(True))
        self.block3 = _make_torchsparse_res_stack(in_channels=p[1], out_channels=p[2], blocks=l[2])
        self.down4 = nn.Sequential(spnn.Conv3d(p[2], p[2], kernel_size=2, stride=2, generative=False), spnn.BatchNorm(p[2]), spnn.ReLU(True))
        self.block4 = _make_torchsparse_res_stack(in_channels=p[2], out_channels=p[3], blocks=l[3])
        self.up4 = nn.Sequential(spnn.Conv3d(p[3], p[4], kernel_size=2, stride=2, transposed=True, generative=False), spnn.BatchNorm(p[4]), spnn.ReLU(True))
        self.block5 = _make_torchsparse_res_stack(in_channels=p[4] + p[2], out_channels=p[4], blocks=l[4])
        self.up3 = nn.Sequential(spnn.Conv3d(p[4], p[5], kernel_size=2, stride=2, transposed=True, generative=False), spnn.BatchNorm(p[5]), spnn.ReLU(True))
        self.block6 = _make_torchsparse_res_stack(in_channels=p[5] + p[1], out_channels=p[5], blocks=l[5])
        self.up2 = nn.Sequential(spnn.Conv3d(p[5], p[6], kernel_size=2, stride=2, transposed=True, generative=False), spnn.BatchNorm(p[6]), spnn.ReLU(True))
        self.block7 = _make_torchsparse_res_stack(in_channels=p[6] + p[0], out_channels=p[6], blocks=l[6])
        self.up1 = nn.Sequential(spnn.Conv3d(p[6], p[7], kernel_size=2, stride=2, transposed=True, generative=False), spnn.BatchNorm(p[7]), spnn.ReLU(True))
        self.block8 = _make_torchsparse_res_stack(in_channels=p[7] + p[0], out_channels=p[7], blocks=l[7])
        self.out_channels = p[7]

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._torchsparse.SparseTensor(voxel_features, voxel_coords, spatial_range=(int(batch_size), *self.sparse_shape))
        out_p1 = self.stem(x)
        out_b1p2 = self.block1(self.down1(out_p1))
        out_b2p4 = self.block2(self.down2(out_b1p2))
        out_b3p8 = self.block3(self.down3(out_b2p4))
        out = self.block4(self.down4(out_b3p8))
        out = self.block5(self._torchsparse.cat([self.up4(out), out_b3p8]))
        out = self.block6(self._torchsparse.cat([self.up3(out), out_b2p4]))
        out = self.block7(self._torchsparse.cat([self.up2(out), out_b1p2]))
        out = self.block8(self._torchsparse.cat([self.up1(out), out_p1]))
        return out


def _make_spconv_res_stack(*, in_channels: int, out_channels: int, blocks: int, indice_key: str):
    layers = [_SpconvResBlock(in_channels, out_channels, indice_key=indice_key)]
    for _ in range(max(0, int(blocks) - 1)):
        layers.append(_SpconvResBlock(out_channels, out_channels, indice_key=indice_key))
    return nn.Sequential(*layers)


class SpconvMinkUNetBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig) -> None:
        super().__init__()
        import spconv.pytorch as spconv

        p = MINKUNET_PLANES
        l = MINKUNET_LAYERS
        self._spconv = spconv
        self.sparse_shape = data_cfg.sparse_shape_zyx
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(model_cfg.input_channels, p[0], 3, padding=1, bias=False, indice_key="stem"),
            nn.BatchNorm1d(p[0]),
            nn.ReLU(),
            spconv.SubMConv3d(p[0], p[0], 3, padding=1, bias=False, indice_key="stem"),
            nn.BatchNorm1d(p[0]),
            nn.ReLU(),
        )
        self.down1 = spconv.SparseSequential(spconv.SparseConv3d(p[0], p[0], 2, stride=2, padding=0, bias=False, indice_key="m1"), nn.BatchNorm1d(p[0]), nn.ReLU())
        self.block1 = _make_spconv_res_stack(in_channels=p[0], out_channels=p[0], blocks=l[0], indice_key="b1")
        self.down2 = spconv.SparseSequential(spconv.SparseConv3d(p[0], p[0], 2, stride=2, padding=0, bias=False, indice_key="m2"), nn.BatchNorm1d(p[0]), nn.ReLU())
        self.block2 = _make_spconv_res_stack(in_channels=p[0], out_channels=p[1], blocks=l[1], indice_key="b2")
        self.down3 = spconv.SparseSequential(spconv.SparseConv3d(p[1], p[1], 2, stride=2, padding=0, bias=False, indice_key="m3"), nn.BatchNorm1d(p[1]), nn.ReLU())
        self.block3 = _make_spconv_res_stack(in_channels=p[1], out_channels=p[2], blocks=l[2], indice_key="b3")
        self.down4 = spconv.SparseSequential(spconv.SparseConv3d(p[2], p[2], 2, stride=2, padding=0, bias=False, indice_key="m4"), nn.BatchNorm1d(p[2]), nn.ReLU())
        self.block4 = _make_spconv_res_stack(in_channels=p[2], out_channels=p[3], blocks=l[3], indice_key="b4")
        self.up4 = spconv.SparseSequential(spconv.SparseInverseConv3d(p[3], p[4], 2, indice_key="m4", bias=False), nn.BatchNorm1d(p[4]), nn.ReLU())
        self.block5 = _make_spconv_res_stack(in_channels=p[4] + p[2], out_channels=p[4], blocks=l[4], indice_key="u4")
        self.up3 = spconv.SparseSequential(spconv.SparseInverseConv3d(p[4], p[5], 2, indice_key="m3", bias=False), nn.BatchNorm1d(p[5]), nn.ReLU())
        self.block6 = _make_spconv_res_stack(in_channels=p[5] + p[1], out_channels=p[5], blocks=l[5], indice_key="u3")
        self.up2 = spconv.SparseSequential(spconv.SparseInverseConv3d(p[5], p[6], 2, indice_key="m2", bias=False), nn.BatchNorm1d(p[6]), nn.ReLU())
        self.block7 = _make_spconv_res_stack(in_channels=p[6] + p[0], out_channels=p[6], blocks=l[6], indice_key="u2")
        self.up1 = spconv.SparseSequential(spconv.SparseInverseConv3d(p[6], p[7], 2, indice_key="m1", bias=False), nn.BatchNorm1d(p[7]), nn.ReLU())
        self.block8 = _make_spconv_res_stack(in_channels=p[7] + p[0], out_channels=p[7], blocks=l[7], indice_key="u1")
        self.out_channels = p[7]

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        x = self._spconv.SparseConvTensor(voxel_features, voxel_coords, list(self.sparse_shape), int(batch_size))
        out_p1 = self.stem(x)
        out_b1p2 = self.block1(self.down1(out_p1))
        out_b2p4 = self.block2(self.down2(out_b1p2))
        out_b3p8 = self.block3(self.down3(out_b2p4))
        out = self.block4(self.down4(out_b3p8))
        out = self.up4(out)
        out = out.replace_feature(torch.cat((out.features, out_b3p8.features), dim=1))
        out = self.block5(out)
        out = self.up3(out)
        out = out.replace_feature(torch.cat((out.features, out_b2p4.features), dim=1))
        out = self.block6(out)
        out = self.up2(out)
        out = out.replace_feature(torch.cat((out.features, out_b1p2.features), dim=1))
        out = self.block7(out)
        out = self.up1(out)
        out = out.replace_feature(torch.cat((out.features, out_p1.features), dim=1))
        out = self.block8(out)
        return out


def _make_minkowski_res_stack(*, in_channels: int, out_channels: int, blocks: int):
    layers = [_MinkowskiResBlock(in_channels, out_channels)]
    for _ in range(max(0, int(blocks) - 1)):
        layers.append(_MinkowskiResBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class MinkowskiMinkUNetBackbone(nn.Module):
    def __init__(self, data_cfg: KittiSecondDataConfig, model_cfg: SemanticKITTISparseResUNet42ModelConfig) -> None:
        super().__init__()
        import MinkowskiEngine as ME

        p = MINKUNET_PLANES
        l = MINKUNET_LAYERS
        self._ME = ME
        self.stem = nn.Sequential(
            ME.MinkowskiConvolution(model_cfg.input_channels, p[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(p[0]),
            ME.MinkowskiReLU(),
            ME.MinkowskiConvolution(p[0], p[0], kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(p[0]),
            ME.MinkowskiReLU(),
        )
        self.down1 = nn.Sequential(ME.MinkowskiConvolution(p[0], p[0], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(p[0]), ME.MinkowskiReLU())
        self.block1 = _make_minkowski_res_stack(in_channels=p[0], out_channels=p[0], blocks=l[0])
        self.down2 = nn.Sequential(ME.MinkowskiConvolution(p[0], p[0], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(p[0]), ME.MinkowskiReLU())
        self.block2 = _make_minkowski_res_stack(in_channels=p[0], out_channels=p[1], blocks=l[1])
        self.down3 = nn.Sequential(ME.MinkowskiConvolution(p[1], p[1], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(p[1]), ME.MinkowskiReLU())
        self.block3 = _make_minkowski_res_stack(in_channels=p[1], out_channels=p[2], blocks=l[2])
        self.down4 = nn.Sequential(ME.MinkowskiConvolution(p[2], p[2], kernel_size=2, stride=2, dimension=3), ME.MinkowskiBatchNorm(p[2]), ME.MinkowskiReLU())
        self.block4 = _make_minkowski_res_stack(in_channels=p[2], out_channels=p[3], blocks=l[3])
        self.up4 = nn.Sequential(ME.MinkowskiConvolutionTranspose(p[3], p[4], kernel_size=2, stride=2, expand_coordinates=False, dimension=3), ME.MinkowskiBatchNorm(p[4]), ME.MinkowskiReLU())
        self.block5 = _make_minkowski_res_stack(in_channels=p[4] + p[2], out_channels=p[4], blocks=l[4])
        self.up3 = nn.Sequential(ME.MinkowskiConvolutionTranspose(p[4], p[5], kernel_size=2, stride=2, expand_coordinates=False, dimension=3), ME.MinkowskiBatchNorm(p[5]), ME.MinkowskiReLU())
        self.block6 = _make_minkowski_res_stack(in_channels=p[5] + p[1], out_channels=p[5], blocks=l[5])
        self.up2 = nn.Sequential(ME.MinkowskiConvolutionTranspose(p[5], p[6], kernel_size=2, stride=2, expand_coordinates=False, dimension=3), ME.MinkowskiBatchNorm(p[6]), ME.MinkowskiReLU())
        self.block7 = _make_minkowski_res_stack(in_channels=p[6] + p[0], out_channels=p[6], blocks=l[6])
        self.up1 = nn.Sequential(ME.MinkowskiConvolutionTranspose(p[6], p[7], kernel_size=2, stride=2, expand_coordinates=False, dimension=3), ME.MinkowskiBatchNorm(p[7]), ME.MinkowskiReLU())
        self.block8 = _make_minkowski_res_stack(in_channels=p[7] + p[0], out_channels=p[7], blocks=l[7])
        self.out_channels = p[7]

    def forward(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        del batch_size
        x = self._ME.SparseTensor(voxel_features, coordinates=voxel_coords)
        out_p1 = self.stem(x)
        out_b1p2 = self.block1(self.down1(out_p1))
        out_b2p4 = self.block2(self.down2(out_b1p2))
        out_b3p8 = self.block3(self.down3(out_b2p4))
        out = self.block4(self.down4(out_b3p8))
        out = self.block5(self._ME.cat(self.up4(out), out_b3p8))
        out = self.block6(self._ME.cat(self.up3(out), out_b2p4))
        out = self.block7(self._ME.cat(self.up2(out), out_b1p2))
        out = self.block8(self._ME.cat(self.up1(out), out_p1))
        return out


def create_sparse_backbone(
    backend: str,
    *,
    data_cfg: KittiSecondDataConfig,
    model_cfg: SemanticKITTISparseResUNet42ModelConfig,
    sorted: bool = False,
):
    name = str(backend).lower()
    if name == "gtsparse":
        return GeometricTemplateMinkUNetBackbone(data_cfg, model_cfg, sorted=bool(sorted))
    if name == "torchsparse":
        return TorchSparseMinkUNetBackbone(data_cfg, model_cfg)
    if name == "spconv":
        return SpconvMinkUNetBackbone(data_cfg, model_cfg)
    if name == "minkowski":
        return MinkowskiMinkUNetBackbone(data_cfg, model_cfg)
    raise KeyError(f"unsupported backend {backend!r}")


def _sparse_features_and_coords(x, backend: str) -> tuple[torch.Tensor, torch.Tensor]:
    name = str(backend).lower()
    if name == "gtsparse":
        return x.features, x.indices
    if name == "torchsparse":
        return x.feats, x.coords
    if name == "spconv":
        return x.features, x.indices
    if name == "minkowski":
        return x.features, x.coordinates
    raise KeyError(f"unsupported backend {backend!r}")


class SemanticKITTISparseResUNet42Model(nn.Module):
    def __init__(self, *, backend: str, config: SemanticKITTISparseResUNet42Config | None = None, sorted: bool = False) -> None:
        super().__init__()
        self.config = config if config is not None else SemanticKITTISparseResUNet42Config()
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
        self.classifier = nn.Linear(int(self.sparse_backbone.out_channels), int(self.config.model.num_classes))

    def encode_batch(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        if isinstance(batch, KittiSecondBatch):
            voxels, voxel_coords, voxel_num_points, batch_size = batch.voxels, batch.voxel_coords, batch.voxel_num_points, int(batch.batch_size)
        else:
            voxels, voxel_coords, voxel_num_points, batch_size = batch["voxels"], batch["voxel_coords"], batch["voxel_num_points"], int(batch["batch_size"])
        return self.vfe(voxels, voxel_num_points), voxel_coords, batch_size

    def forward_sparse_backbone(self, voxel_features: torch.Tensor, voxel_coords: torch.Tensor, batch_size: int):
        return self.sparse_backbone(voxel_features, voxel_coords, batch_size)

    def forward_sparse_backbone_from_batch(self, batch: KittiSecondBatch | dict[str, torch.Tensor]):
        voxel_features, voxel_coords, batch_size = self.encode_batch(batch)
        return self.forward_sparse_backbone(voxel_features, voxel_coords, batch_size)

    def forward(self, batch: KittiSecondBatch | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sparse_out = self.forward_sparse_backbone_from_batch(batch)
        feats, coords = _sparse_features_and_coords(sparse_out, self.backend)
        logits = self.classifier(feats)
        return {
            "logits": logits,
            "coords": coords,
        }

    def predict(self, predictions: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.argmax(predictions["logits"], dim=1)


def _log_dir_for_run(*, root_log_dir: Path, device: str, model: SemanticKITTISparseResUNet42Model, data_root: Path, sweeps: int) -> Path:
    gpu_model = _sanitize_path_component(torch.cuda.get_device_name(torch.device(device).index or torch.cuda.current_device()))
    dtype_name = _sanitize_path_component(str(next(model.parameters()).dtype).replace("torch.", ""))
    dataset_name = _sanitize_path_component(data_root.name)
    return Path(root_log_dir) / f"logs_{gpu_model}_{dtype_name}_minkunet_{dataset_name}_sweeps{int(sweeps)}"


def _append_backend_log_frame(log_file, record: dict[str, object]) -> None:
    payload = {
        "batch_index": int(record["batch_index"]),
        "frame_ids": list(record["frame_ids"]),
        "end2end_ms": float(record["end2end_ms"]),
        "timing_repeats": int(record.get("timing_repeats", 1)),
        "timing_warmup_repeats": int(record.get("timing_warmup_repeats", 0)),
        "logit_rows": int(record["logit_rows"]),
        "num_classes": int(record["num_classes"]),
    }
    json.dump(payload, log_file, ensure_ascii=True, sort_keys=True)
    log_file.write("\n")
    log_file.flush()


def _measure_frame_timings(
    model: SemanticKITTISparseResUNet42Model,
    loader,
    *,
    device: str,
    warmup: int,
    timing_repeats: int,
    timing_warmup_repeats: int,
    on_result=None,
):
    resolved_device = require_cuda_device(device)
    runtime_dtype = next(model.parameters()).dtype
    from .kitti_second import _iter_device_batches

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
    measured_batches = tqdm(
        device_batches,
        total=len(loader),
        desc=f"minkunet/semantickitti/{model.backend}",
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(measured_batches):
            predictions, end2end_ms = measure_cuda_elapsed_ms(
                model,
                batch,
                device=resolved_device,
                repeats=max(1, int(timing_repeats)),
                warmup_repeats=max(0, int(timing_warmup_repeats)),
                clear_metadata=clear_metadata,
            )
            record = {
                "batch_index": int(batch_index),
                "frame_ids": list(batch.frame_ids),
                "end2end_ms": float(end2end_ms),
                "timing_repeats": int(max(1, timing_repeats)),
                "timing_warmup_repeats": int(max(0, timing_warmup_repeats)),
                "logit_rows": int(predictions["logits"].size(0)),
                "num_classes": int(predictions["logits"].size(1)),
            }
            results.append(record)
            if on_result is not None:
                on_result(record)
            del predictions
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SemanticKITTI + MinkUNet thin runner")
    parser.add_argument("--backend", type=str, default="gtsparse")
    parser.add_argument("--dtype", type=str, default="fp32", choices=("fp32", "fp16"))
    parser.add_argument("--sorted", action="store_true", help="Enable sorted sparse-conv variants for GTSparse and TorchSparse backends.")
    parser.add_argument("--spconv-disable-sort", action="store_true", help="Set spconv SPCONV_DO_SORT=False before creating the spconv backbone.")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/semantickitti"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--frame", type=str, default="")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--timing-warmup-repeats", type=int, default=2)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--sweeps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> dict[str, object]:
    runtime_dtype = resolve_runtime_dtype(args.dtype)
    require_cuda_device(args.device)
    if str(args.backend).lower() == "minkowski" and runtime_dtype == torch.float16:
        raise ValueError("Minkowski backend does not support fp16 in this runner")
    _configure_spconv_do_sort(not bool(getattr(args, "spconv_disable_sort", False)))
    config = SemanticKITTISparseResUNet42Config(
        data=SemanticKITTISparseResUNet42DataConfig(
            root=discover_semantickitti_root(args.data_root),
            split=str(args.split),
            max_sweeps=max(1, int(getattr(args, "sweeps", 1))),
        )
    )
    model = SemanticKITTISparseResUNet42Model(
        backend=args.backend,
        config=config,
        sorted=bool(getattr(args, "sorted", False)),
    ).to(args.device)
    if runtime_dtype == torch.float16:
        model = model.half()
    model.eval()
    dataset = _make_semantickitti_dataset(config.data)
    indices = _iter_sample_indices(dataset, frame_id=str(args.frame), num_samples=int(args.frames))
    loader = _make_loader(dataset, indices, config.data, batch_size=int(args.batch))
    warmup_batches = min(int(args.warmup), len(loader))
    log_dir = _log_dir_for_run(root_log_dir=Path(getattr(args, "log_dir", Path("logs"))), device=str(args.device), model=model, data_root=config.data.root, sweeps=int(config.data.max_sweeps))
    log_path = log_dir / f"{args.backend}.jsonl"
    config_path = log_dir / f"{args.backend}.config.json"
    summary_path = log_dir / f"{args.backend}.summary.json"
    device_index = torch.device(args.device).index
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device() if device_index is None else device_index)
    workload = "minkunet_semantickitti_sweeps1"
    _write_json_file(
        config_path,
        {
            "backend": str(args.backend),
            "batch": int(args.batch),
            "data_root": str(config.data.root),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "sorted": bool(getattr(args, "sorted", False)),
            "spconv_do_sort": not bool(getattr(args, "spconv_disable_sort", False)),
            "frame": str(args.frame),
            "frames": int(args.frames),
            "gpu": gpu_name,
            "log_dir": str(log_dir),
            "split": str(args.split),
            "sweeps": int(config.data.max_sweeps),
            "timing_repeats": int(max(1, args.timing_repeats)),
            "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
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
            on_result=lambda record: _append_backend_log_frame(log_file, record),
        )
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
            "split": str(args.split),
            "spconv_do_sort": not bool(getattr(args, "spconv_disable_sort", False)),
            "sweeps": int(config.data.max_sweeps),
            "stats": {"end2end": _stats_dict(end2end_times)},
            "timing_repeats": int(max(1, args.timing_repeats)),
            "timing_warmup_repeats": int(max(0, args.timing_warmup_repeats)),
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
        "spconv_do_sort": not bool(getattr(args, "spconv_disable_sort", False)),
        "sweeps": int(config.data.max_sweeps),
        "frames": int(len(indices)),
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
    end2end_times = [float(record['end2end_ms']) for record in summary['results']]
    if end2end_times:
        print(_format_stats("end2end", end2end_times))
    print("=" * 60)


if __name__ == "__main__":
    main()
