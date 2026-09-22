#!/usr/bin/env bash

if [[ -n "${ZSH_VERSION:-}" ]]; then
  gtsparse_activate="${(%):-%x}"
else
  gtsparse_activate="${BASH_SOURCE[0]}"
fi

gtsparse_root="$(cd "$(dirname "$gtsparse_activate")/.." && pwd)"
source "$gtsparse_root/.venv/bin/activate"

if [[ -x "$gtsparse_root/.cuda/cuda-12.1/bin/nvcc" ]]; then
  export CUDA_HOME="$gtsparse_root/.cuda/cuda-12.1"
else
  export CUDA_HOME=/usr/local/cuda-12.1
fi

export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CPATH="$gtsparse_root/.venv/include${CPATH:+:$CPATH}"
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
