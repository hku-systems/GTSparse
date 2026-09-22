import os
import re
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch


ROOT = Path(__file__).resolve().parent
GTSPARSE_CUDA_DIR = ROOT / "src" / "gtsparse" / "cuda"

os.environ.setdefault("MAX_JOBS", "4")

BUILD_MODE = os.environ.get("BUILD_MODE", "development").lower()


def get_compile_args():
    if BUILD_MODE == "production":
        cxx_args = [
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-march=native",
            "-mtune=native",
            "-ffast-math",
            "-funroll-loops",
            "-ftree-vectorize",
            "-fomit-frame-pointer",
        ]
        nvcc_args = [
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "--use_fast_math",
            "--ptxas-options=-v",
            "--restrict",
        ]
    elif BUILD_MODE == "debug":
        cxx_args = [
            "-std=c++17",
            "-O0",
            "-g",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-fsanitize=address",
            "-fno-omit-frame-pointer",
        ]
        nvcc_args = [
            "-std=c++17",
            "-O0",
            "-g",
            "-G",
            "--device-debug",
            "--generate-line-info",
        ]
    else:
        cxx_args = [
            "-std=c++17",
            "-O2",
            "-g",
            "-fPIC",
            "-Wall",
            "-Wextra",
        ]
        nvcc_args = [
            "-std=c++17",
            "-O2",
            "-g",
            "--generate-line-info",
        ]

    return cxx_args, nvcc_args


def get_cuda_arch_args():
    cuda_arch = os.environ.get("CUDA_ARCH", "").strip()
    if cuda_arch:
        arch_args = []
        for item in cuda_arch.replace(";", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if item.startswith("-gencode"):
                arch_args.append(item)
                continue
            digits = item.replace(".", "")
            arch_args.append(f"-gencode=arch=compute_{digits},code=sm_{digits}")
        return arch_args

    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            compute = f"{major}{minor}"
            return [f"-gencode=arch=compute_{compute},code=sm_{compute}"]
    except Exception as exc:
        print(f"Warning: Could not detect GPU architecture: {exc}")

    return [
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-gencode=arch=compute_89,code=sm_89",
        "-gencode=arch=compute_90,code=sm_90",
    ]


def sparse_cnn_sources():
    return [
        str(GTSPARSE_CUDA_DIR / "pybind.cpp"),
        str(GTSPARSE_CUDA_DIR / "simt_fp32.cu"),
        str(GTSPARSE_CUDA_DIR / "tc_fp16.cu"),
        str(GTSPARSE_CUDA_DIR / "kernel3_fp32.cu"),
        str(GTSPARSE_CUDA_DIR / "kernel3_fp16.cu"),
        str(GTSPARSE_CUDA_DIR / "kernel9_fp32.cu"),
        str(GTSPARSE_CUDA_DIR / "kernel9_fp16.cu"),
        str(GTSPARSE_CUDA_DIR / "kernel8_fp32.cu"),
        str(GTSPARSE_CUDA_DIR / "kernel8_fp16.cu"),
        str(GTSPARSE_CUDA_DIR / "entry.cu"),
        str(GTSPARSE_CUDA_DIR / "build_runtime_from_coords.cu"),
        str(GTSPARSE_CUDA_DIR / "build_runtime_from_dense_out_in_map.cu"),
        str(GTSPARSE_CUDA_DIR / "build_full_runtime_from_coords.cu"),
        str(GTSPARSE_CUDA_DIR / "build_reverse_runtime_from_coords.cu"),
        str(GTSPARSE_CUDA_DIR / "build_reverse_runtime_from_full_runtime.cu"),
        str(GTSPARSE_CUDA_DIR / "build_full_runtime_k3.cu"),
        str(GTSPARSE_CUDA_DIR / "build_runtime_k9.cu"),
        str(GTSPARSE_CUDA_DIR / "build_runtime_k8.cu"),
    ]
def get_target_sms(arch_args):
    sms = []
    for arg in arch_args:
        match = re.search(r"sm_(\d+)", arg)
        if match:
            sms.append(int(match.group(1)))
    return sms


cxx_args, nvcc_args = get_compile_args()
cuda_arch_args = get_cuda_arch_args()
target_sms = get_target_sms(cuda_arch_args)
has_fp16_tc = any(sm >= 70 for sm in target_sms)
has_tf32_tc = any(sm >= 80 for sm in target_sms)
tc_macros = [
    f"-DGTSPARSE_FP16_TC_ENABLED={1 if has_fp16_tc else 0}",
    f"-DGTSPARSE_TF32_TC_ENABLED={1 if has_tf32_tc else 0}",
]
cxx_args = cxx_args + tc_macros
nvcc_args = nvcc_args + cuda_arch_args + tc_macros

print(f"Building GTSparse in {BUILD_MODE} mode")
print(f"CXX args: {' '.join(cxx_args)}")
print(f"NVCC args: {' '.join(nvcc_args)}")
print(f"Tensor Core compile support: FP16={'on' if has_fp16_tc else 'off'} TF32={'on' if has_tf32_tc else 'off'}")


setup(
    name="gtsparse",
    version="0.1.0",
    description="Geometric-Template-Driven Sparse Convolution Runtime on GPUs",
    license="BSD-3-Clause",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="gtsparse._C",
            sources=sparse_cnn_sources(),
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": nvcc_args,
            },
            language="c++17",
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
