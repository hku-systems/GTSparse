#!/usr/bin/env bash
set -euo pipefail

gtsparse_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$gtsparse_root"

sudo apt-get update
sudo apt-get install -y build-essential python3-dev python3-venv libopenblas-dev \
  autoconf automake git curl

python3 -m venv .venv

if [[ ! -x /usr/local/cuda-12.1/bin/nvcc && ! -x .cuda/cuda-12.1/bin/nvcc ]]; then
  bash scripts/install_cuda.sh
fi

source scripts/activate.sh
"$CUDA_HOME/bin/nvcc" --version | grep 'release 12.1'

python -m pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0 ninja
python -m pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

if gtsparse_arch="$(python -c 'import torch; a=torch.cuda.get_device_capability(); print(f"{a[0]}.{a[1]}")' 2>/dev/null)"; then
  export CUMM_CUDA_ARCH_LIST="$gtsparse_arch"
  export TORCH_CUDA_ARCH_LIST="$gtsparse_arch"
  export CUDA_ARCH="$gtsparse_arch"
else
  # No GPU at install time: target the platforms the paper evaluates on.
  # Override these variables for other GPUs.
  export CUMM_CUDA_ARCH_LIST="8.0;8.6;8.9"
  export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
  export CUDA_ARCH="8.0;8.6;8.9"
fi
export FORCE_CUDA=1
export MAX_JOBS=4
unset CUMM_CUDA_VERSION

(
  cd third_parties/sparsehash
  ./configure --prefix="$VIRTUAL_ENV"
  make -j4
  make install
)

(
  cd third_parties/cumm
  python -m pip install -e .
)
CUMM_DISABLE_JIT=0 python -c 'import cumm'

(
  cd third_parties/spconv
  python -m pip install --no-build-isolation -e .
)
CUMM_DISABLE_JIT=1 SPCONV_DISABLE_JIT=0 python -c 'import spconv.pytorch'

(
  cd third_parties/torchsparse
  python -m pip install --no-build-isolation -e .
)
python -c 'import torchsparse'

(
  cd third_parties/MinkowskiEngineCuda13
  python setup.py install --blas=openblas
)
python -c 'import MinkowskiEngine'

BUILD_MODE=production python -m pip install --no-build-isolation -e .
if python -c 'import torch; exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
  python scripts/validate_install.py
else
  echo "no GPU in build environment; skipping validate_install.py (run it inside the container with --gpus all)"
fi
