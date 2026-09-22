#!/usr/bin/env bash
set -euo pipefail

gtsparse_root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$gtsparse_root/.cache" "$gtsparse_root/.cuda"

curl -L \
  https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run \
  -o "$gtsparse_root/.cache/cuda_12.1.run"

sh "$gtsparse_root/.cache/cuda_12.1.run" \
  --silent \
  --toolkit \
  --toolkitpath="$gtsparse_root/.cuda/cuda-12.1" \
  --override
