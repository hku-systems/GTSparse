#!/usr/bin/env bash
set -euo pipefail

workspace=/workspace/GTSparse
marker="$workspace/.gtsparse-installed"

if [[ ! -f "$marker" ]]; then
  echo "first run: copying sources and installing (needs GPU access, ~40-60 minutes)"
  mkdir -p "$workspace"
  cp -a /opt/gtsparse-src/. "$workspace/"
  cd "$workspace"
  bash scripts/install.sh
  touch "$marker"
  # The dedicated install command installs once and exits; any other first
  # command (e.g., an experiment or `sleep infinity`) continues after installing.
  if [[ "${@: -1}" == "scripts/install.sh" ]]; then
    echo "installation complete"
    exit 0
  fi
fi

cd "$workspace"
source scripts/activate.sh
exec "$@"
