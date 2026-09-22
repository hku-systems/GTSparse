# GTSparse: A Geometric-Template-Driven Sparse Convolution Runtime on GPUs (Artifact Evaluation)

## Overview

This artifact reproduces the experiments in the paper. It builds GTSparse and three baseline sparse convolution engines (SpConv v2, TorchSparse++, MinkowskiEngine) from source, runs end-to-end inference, microbenchmark, ablation, and sensitivity experiments, and regenerates the paper's tables and figures from the raw measurements.

Tested environment: Ubuntu 22.04, NVIDIA driver 560+, CUDA Toolkit 12.1, Python 3.10, PyTorch 2.1.2. The reference machine has an Intel i7-10700 (32 GB RAM) with an NVIDIA GeForce RTX 3080 (10 GB). Any Linux machine with a supported NVIDIA GPU (sm 70+, 8+ GB VRAM) and at least 400 GB of free disk space should work.

Permanent archive: [10.5281/zenodo.22879641](https://doi.org/10.5281/zenodo.22879641)

MinkowskiEngine may print an `OMP_NUM_THREADS` warning on first import; this is harmless and can be ignored.

## Dataset Preparation

Download KITTI, NuScenes, and SemanticKITTI from the official sources:
- [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d)
- [NuScenes](https://www.nuscenes.org/nuscenes)
- [SemanticKITTI](http://semantic-kitti.org/)

Organize them as follows (create a `dataset` directory and place the datasets in it):

```
dataset
├── kitti
│   ├── testing
│   └── training
├── nuscenes
│   ├── LICENSE
│   ├── maps
│   ├── samples
│   ├── sweeps
│   └── v1.0-test
└── semantickitti
    ├── README
    └── dataset
        └── sequences
```

Users are responsible for complying with the original licenses of KITTI, NuScenes, and SemanticKITTI.

Note: the nuScenes loader probes `v1.0-trainval`, then `v1.0-mini`, then `v1.0-test` under the given root, so different versions (e.g., mini and test) must be placed under separate roots.

## Installation

Organize them as follows (create a `dataset` directory and place the datasets in it):

The artifact uses Python 3.10, PyTorch 2.1.2, and CUDA Toolkit 12.1. Clone the baseline submodules and install:

```bash
git submodule update --init --recursive
bash scripts/install.sh
source scripts/activate.sh
```

The installer creates `.venv`, installs the Ubuntu build dependencies (via `sudo apt-get`; set `DEBIAN_FRONTEND=noninteractive` for unattended installs), installs PyTorch cu121, and builds cumm, SpConv, TorchSparse++, MinkowskiEngine, and GTSparse from source. The first SpConv import compiles its generated CUDA sources and can take several minutes. The installer uses `/usr/local/cuda-12.1` when available; if CUDA Toolkit 12.1 is not installed, it downloads the toolkit without a driver under `.cuda/cuda-12.1`.

Run the activation script before evaluating in a new shell:

```bash
source scripts/activate.sh
python scripts/validate_install.py
```

`validate_install.py` runs one 3x3x3 SubM convolution layer through all four engines against a dense PyTorch reference, with TF32 disabled. The comparison is tolerance-based: engines that accumulate in FP32 (GTSparse, TorchSparse++, MinkowskiEngine) agree at FP32 1e-3 and FP16 2e-2 (atol/rtol); SpConv's FP16 kernels accumulate in FP16 and are checked at atol=1.0. Bitwise equality across engines is not expected, since each engine reduces the kernel offsets in a different order and the production build uses fast-math.

### Container

A thin Dockerfile is provided: the image contains only the CUDA toolkit and OS dependencies, and the environment is built inside a named volume on the first run when the GPU is present, matching the bare-metal path (arch detection, validation). Build the image (fast, no GPU needed):

```bash
bash scripts/build_container.sh
```

The environment is installed on the first run (~40–60 minutes; the host needs `nvidia-container-toolkit` for `--gpus all`). The install already runs `validate_install.py` at the end. `--shm-size=1g` is required: the DataLoader workers share batches through `/dev/shm`, whose container default (64 MB) is too small for multi-sweep point clouds. In the commands below, `gtsparse-artifact` is the image built above and `gtsparse-env` is a Docker named volume that persists the installed environment across runs.

```bash
# 1. Install the environment (one time):
docker run --rm --gpus all --shm-size=1g -v gtsparse-env:/workspace \
  gtsparse-artifact bash scripts/install.sh
# 2. Run experiments (reuses the installed environment):
docker run --rm --gpus all --shm-size=1g -v gtsparse-env:/workspace \
  -v /path/to/dataset:/workspace/GTSparse/dataset \
  gtsparse-artifact bash run_artifact.sh --end-to-end fp16
```

Datasets are intentionally not baked into the image; mount them at runtime. The bare-metal installation above remains the primary installation path.

## Evaluation

### Quick Start

Run every experiment and generate all tables and figures:

```bash
bash run_artifact.sh --all
```

Or run by category:

```bash
bash run_artifact.sh --end-to-end fp16
bash run_artifact.sh --end-to-end fp32
bash run_artifact.sh --microbenchmark
bash run_artifact.sh --ablation
bash run_artifact.sh --sensitivity
bash run_artifact.sh --plots            # regenerate tables/figures from existing logs
```

### Example Runs with Expected Outputs

Each category has a small example run that completes in a few minutes. The examples write to `examples/` (via `--output-root`), leaving `logs/` untouched. Reference outputs captured on the RTX 3080 are stored in `examples/logs/`; see `examples/README.md` for the commands and what to compare.

```bash
bash run_artifact.sh --end-to-end fp16 --frames 5 --output-root examples
bash run_artifact.sh --end-to-end fp32 --frames 5 --output-root examples
bash run_artifact.sh --microbenchmark --micro-frames 10 --memory-frames 5 --frames 50 --output-root examples
bash run_artifact.sh --ablation --frames 50 --output-root examples
bash run_artifact.sh --sensitivity --frames 50 --output-root examples
```

### Parameters

Common optional parameters for `run_artifact.sh`:

| Parameter | Default | Description |
|---|---|---|
| `--frames N` | `0` (full split) | Number of frames for end-to-end, template-profile, ablation, and sensitivity |
| `--micro-frames N` | `100` | Sampled frames for timing, profiling, and breakdown |
| `--memory-frames N` | `20` | Frames for peak-memory measurement |
| `--warmup N` | `20` | Global warmup frames before measurement |
| `--timing-repeats N` | `3` | Measured repetitions per frame; the median is recorded |
| `--device DEVICE` | `cuda:0` | CUDA device |
| `--output-root DIR` | `.` | Root directory for logs, results, and figures |
| `--overwrite` | off | Replace completed outputs instead of reusing them |

The environment variable `GTSPARSE_MIN_TEMPLATE` controls the minimum template index for the ablation experiment (`0`, `1`, `4`, or `7`; set automatically by `--ablation`).

### Performance Configuration

Absolute latency can vary across machines even with the same GPU model due to differences in CPU, memory subsystem, PCIe topology, firmware, and power/clock policies. To reduce run-to-run variation, we recommend locking the CPU and GPU clocks and keeping the settings unchanged across all backends. The relative performance trends remain consistent regardless.

The following shows the configuration used on the reference RTX 3080 machine (Intel i7-10700). Suitable values depend on the specific hardware.

```bash
# CPU: performance governor with a sustainable minimum frequency and power limit.
sudo systemctl stop thermald
sudo cpupower frequency-set -g performance
sudo cpupower frequency-set -d 4.60GHz
echo 125000000 | sudo tee /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw

# GPU: persistence mode, sustainable power limit and locked clocks.
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 340
sudo nvidia-smi -lgc 1800,1800
sudo nvidia-smi -lmc 9501,9501  # omit if locked memory clocks are unsupported
nvidia-smi --query-gpu=pstate,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv

# To revert:
# sudo cpupower frequency-set -g powersave
# sudo nvidia-smi -rgc
# sudo nvidia-smi -rmc
# sudo nvidia-smi -pl <default watts>
# sudo systemctl start thermald
```

These settings are optional. Evaluators who cannot lock clocks (e.g., on rented hardware) should judge ratios rather than absolute numbers, which is what the paper's claims are stated in.

### Resource Requirements

Measured on the reference RTX 3080 with the locked clock settings; a faster GPU reduces the GPU time roughly proportionally. Wall time is 10–20% higher than GPU time due to data loading.

| Experiment | Command | GPU time (RTX 3080) | Logs written |
|---|---|---|---|
| End-to-end FP16 | `--end-to-end fp16` | ~1.3 h | ~13 MB |
| End-to-end FP32 | `--end-to-end fp32` | ~0.9 h | ~4 MB |
| Microbenchmark | `--microbenchmark` | ~5 min | ~12 MB |
| Ablation | `--ablation` | ~20 min | ~5 MB |
| Sensitivity | `--sensitivity` | ~2.4 h | ~18 MB |
| All categories | `--all` | ~5 h (~6 h wall) | ~52 MB |

Installation takes about 40 minutes, plus several minutes for SpConv's first-import JIT compilation. The extracted datasets occupy about 200 GB (KITTI ~39 GB, nuScenes ~72 GB, SemanticKITTI ~90 GB); downloading and extracting `dataset.tar` requires roughly twice the extracted size (~400 GB of free space).

### Measurement Methodology

**End-to-end.** FP16 runs GTSparse, SpConv, and TorchSparse++ in FP16 and MinkowskiEngine in FP32 (MinkowskiEngine does not support FP16); FP32 runs all four in FP32. SpConv uses its default sorted bitmask path and TF32 is disabled in all runs. Detection workloads (SECOND, VoxelNeXt) report sparse-convolution latency; the segmentation workload (MinkUNet42) reports full pipeline latency.

**Template distribution.** Family percentages and average width aggregate output rows across all `3x3x3` (`K=27`) layers on the complete split; layers with other kernel volumes are excluded.

**Effective throughput.** Useful and issued work are reconstructed from the first 100 frames of each split in a per-layer profile: useful work is the active-pair count, and issued work is the assigned rows times the actual per-template payload slot count (before launch tile padding). SpConv's issued work uses the M-tile from its RTX 4090 autotune (M=32 for small output counts, M=64 otherwise); pass `--spconv-tile autotuned` to use the M-tile SpConv autotuned on the machine that produced the logs.

**Time breakdown.** Breakdown uses 100 frames for every backend. GTSparse uses native builder/kernel CUDA events; baseline breakdown uses one cold-to-warm pair per independent frame. Aggregation takes the median per-frame component shares and scales them by the overall end-to-end latency, so the displayed builder and kernel values sum to the complete model latency (Table 3 in the paper).

**Peak memory.** Reports `torch.cuda.max_memory_allocated` from model construction through the measured forwards, with each backend run in a separate process.

**Ablation.** Evaluates `min_template=0,1,4,7` on VoxelNeXt with 10 sweeps.

**Sensitivity.** Evaluates all systems with 1, 5, 10, and 20 sweeps.

**Paper tables vs. artifact outputs.** The builder-share table (Table 3), the peak-memory table (Table 4), and the throughput figure (Fig 5) were measured on an RTX 4090 with the development pipeline, while the released logs carry the RTX 3080. Builder shares shift with the platform: on a faster GPU the convolution kernels run faster, so the kernel portion shrinks and the builder's share rises. Peak memory depends on the measurement method and allocator behavior. Raw TFLOPS scale with the GPU. The corresponding claims are stated as shares, orderings, and deltas (30–186 MB), which reproduce across platforms.

### Outputs and Paper Reproduction

The complete reference logs collected on the NVIDIA GeForce RTX 3080 are available in the [`artifact-logs-rtx3080-v1` release](https://github.com/eurosys-27-1183/GTSparse/releases/tag/artifact-logs-rtx3080-v1). From the repository root, download and extract the logs, then regenerate all tables and figures:

```bash
curl -fL https://github.com/eurosys-27-1183/GTSparse/releases/download/artifact-logs-rtx3080-v1/gtsparse-artifact-logs-rtx3080.tar.gz -o gtsparse-artifact-logs-rtx3080.tar.gz && tar -xzf gtsparse-artifact-logs-rtx3080.tar.gz
bash run_artifact.sh --plots
```

Raw measurements are written under `logs/`; microbenchmark cases are separated by GPU, dtype, and workload so measurements from multiple platforms can coexist. Aggregated data is written as CSV files under `results/`. The cross-platform end-to-end figure is written directly under `figures/`, while all platform-specific tables and figures are written under `figures/<gpu>/`. After the measurements are available, `bash run_artifact.sh --plots` regenerates both `results/` and `figures/` directly from the raw logs.

## Citation

If you use this artifact in research, please cite the paper (machine-readable metadata is also provided in `CITATION.cff`):

```bibtex
@inproceedings{gtsparse2027,
  author    = {Guan, Xiuxian and Zhang, Zongyuan and Qi, Ji and Cui, Heming},
  title     = {GTSparse: A Geometric-Template-Driven Sparse Convolution Runtime for GPUs},
  booktitle = {Proceedings of the 22nd European Conference on Computer Systems (EuroSys '27)},
  publisher = {ACM},
  address   = {New York, NY, USA},
  year      = {2027},
  doi       = {10.1145/3842654.3848577},
}
```

## License

GTSparse is licensed under the BSD 3-Clause License (see `LICENSE`), which permits use, comparison, and extension with attribution. The baseline engines under `third_parties/` retain their own licenses: cumm and SpConv are Apache-2.0, TorchSparse++ is MIT, sparsehash is BSD-3-Clause, and MinkowskiEngine (the CUDA-13-compatible fork) is MIT. The datasets are distributed under their respective original licenses; users are responsible for complying with them.