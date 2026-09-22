from pathlib import Path

import torch


WORKLOADS = (
    "second_kitti_sweeps1",
    "voxelnext_nuscenes_sweeps1",
    "voxelnext_nuscenes_sweeps10",
    "minkunet_semantickitti_sweeps1",
)


def build_workload(
    workload: str,
    backend: str,
    dtype: str,
    frames: int,
    device: str,
    *,
    random_sample: bool = False,
):
    from gtsparse.e2e_v2.kitti_second import _iter_sample_indices, _make_loader

    if workload == "second_kitti_sweeps1":
        from gtsparse.e2e_v2.kitti_second import KittiLidarDataset, KittiSecondConfig, KittiSecondDataConfig, KittiSecondModel

        config = KittiSecondConfig(data=KittiSecondDataConfig(root=Path("dataset/kitti"), split="test", max_sweeps=1))
        model = KittiSecondModel(backend=backend, config=config)
        dataset = KittiLidarDataset(config.data.root, split=config.data.split)
    elif workload.startswith("voxelnext_nuscenes_sweeps"):
        from gtsparse.e2e_v2.nuscenes_voxelnext import (
            NuScenesVoxelNeXtConfig,
            NuScenesVoxelNeXtDataConfig,
            NuScenesVoxelNeXtModel,
            _configure_torchsparse_backend_for_nuscenes,
            _make_nuscenes_dataset,
        )

        sweeps = int(workload.rsplit("sweeps", 1)[1])
        if backend == "torchsparse":
            _configure_torchsparse_backend_for_nuscenes()
        config = NuScenesVoxelNeXtConfig(
            data=NuScenesVoxelNeXtDataConfig(root=Path("dataset/nuscenes"), split="test", max_sweeps=sweeps)
        )
        model = NuScenesVoxelNeXtModel(backend=backend, config=config)
        dataset = _make_nuscenes_dataset(config.data)
    elif workload == "minkunet_semantickitti_sweeps1":
        from gtsparse.e2e_v2.semantickitti_sparse_resunet42 import (
            SemanticKITTISparseResUNet42Config,
            SemanticKITTISparseResUNet42DataConfig,
            SemanticKITTISparseResUNet42Model,
            _configure_spconv_do_sort,
            _make_semantickitti_dataset,
        )

        _configure_spconv_do_sort(True)
        config = SemanticKITTISparseResUNet42Config(
            data=SemanticKITTISparseResUNet42DataConfig(
                root=Path("dataset/semantickitti"), split="val", max_sweeps=1
            )
        )
        model = SemanticKITTISparseResUNet42Model(backend=backend, config=config)
        dataset = _make_semantickitti_dataset(config.data)
    else:
        raise ValueError(f"unknown workload {workload}")

    model = model.to(device)
    runtime_dtype = torch.float16 if dtype == "fp16" else torch.float32
    if runtime_dtype == torch.float16:
        model = model.half()
    model.eval()
    indices = _iter_sample_indices(
        dataset,
        frame_id="",
        num_samples=int(frames),
        random_sample=random_sample,
    )
    loader = _make_loader(dataset, indices, config.data, batch_size=1)
    return model, loader, runtime_dtype


def move_batch(batch, device: str, dtype: torch.dtype):
    from gtsparse.e2e_v2.kitti_second import move_kitti_second_batch_to_device

    return move_kitti_second_batch_to_device(batch, device, dtype=dtype)


def sparse_forward(model, batch, workload: str):
    if workload == "minkunet_semantickitti_sweeps1":
        return model(batch)
    voxel_features, voxel_coords, batch_size = model.encode_batch(batch)
    return model.forward_sparse_convolutions(voxel_features, voxel_coords, batch_size)
