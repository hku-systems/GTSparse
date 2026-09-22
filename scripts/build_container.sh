#!/usr/bin/env bash
set -euo pipefail

image_name="${IMAGE_NAME:-gtsparse-artifact}"

gtsparse_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$gtsparse_root"

if [[ ! -f Dockerfile ]]; then
  echo "error: Dockerfile not found in $gtsparse_root" >&2
  exit 1
fi

# Baseline sources must be present in the build context. Tolerate hosts where
# the submodule metadata is unavailable but the sources are already checked out.
if ! git submodule update --init --recursive; then
  echo "warning: submodule update failed; using existing third_parties contents" >&2
fi

docker build -t "$image_name" .

echo "Built $image_name (thin image; the environment is installed on first run)."
echo "1. Install the environment (one time, needs GPU access; --shm-size=1g is"
echo "   required for the DataLoader workers):"
echo "  docker run --rm --gpus all --shm-size=1g -v gtsparse-workspace:/workspace \\"
echo "    $image_name bash scripts/install.sh"
echo "2. Run experiments:"
echo "  docker run --rm --gpus all --shm-size=1g -v gtsparse-workspace:/workspace \\"
echo "    -v /path/to/dataset:/workspace/GTSparse/dataset $image_name bash run_artifact.sh --end-to-end fp16"
