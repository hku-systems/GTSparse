# GTSparse artifact container: a thin image; the environment is built inside
# a named volume on the first run, when the GPU is present (--gpus all), so
# installation matches the bare-metal path (arch detection + validation).
#
# Build the image (fast, no GPU needed):
#   bash scripts/build_container.sh
#
# First run installs the environment (~40-60 minutes, needs --gpus all):
#   docker run --gpus all -v gtsparse-workspace:/workspace \
#       -v /path/to/dataset:/workspace/GTSparse/dataset \
#       gtsparse-artifact bash run_artifact.sh --end-to-end fp16
#
# Subsequent runs reuse the installed environment in the volume.
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# OS dependencies must live in the image: the venv built at first run keeps
# symlinks into /usr (python3, libopenblas for MinkowskiEngine), which would
# dangle if these packages only existed in the first container's ephemeral layer.
# install.sh calls `sudo apt-get`; provide sudo so it works as root.
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo ca-certificates build-essential python3-dev python3-venv \
    libopenblas-dev autoconf automake git curl && \
    rm -rf /var/lib/apt/lists/*

COPY . /opt/gtsparse-src
COPY scripts/container_entrypoint.sh /usr/local/bin/gtsparse-entrypoint
RUN chmod +x /usr/local/bin/gtsparse-entrypoint

ENTRYPOINT ["gtsparse-entrypoint"]
CMD ["bash"]
