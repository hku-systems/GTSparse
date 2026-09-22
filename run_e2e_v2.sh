#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
backend="${1:?usage: ./run_e2e_v2.sh <backend> <dtype> [frames] [warmup_batches]}"
dtype="${2:?usage: ./run_e2e_v2.sh <backend> <dtype> [frames] [warmup_batches]}"
frames="${3:-0}"
warmup="${4:-20}"
repeats="${5:-3}"
timing_warmup="${6:-2}"
device="${7:-cuda:0}"
log_dir="${8:-logs/end_to_end}"
overwrite="${9:-0}"
gpu_key="$(python -c 'import torch; s=torch.cuda.get_device_name(torch.device("'"$device"'")); print("".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_"))')"
run_case() {
  local module="$1" data_root="$2" split="$3" sweeps="$4" model="$5" run_dtype="$dtype"
  [[ "$backend" == "minkowski" ]] && run_dtype=fp32
  local dtype_label="float16"
  [[ "$run_dtype" == "fp32" ]] && dtype_label="float32"
  local summary_path="$log_dir/logs_${gpu_key}_${dtype_label}_${model}_$(basename "$data_root")_sweeps${sweeps}/${backend}.summary.json"
  if [[ "$overwrite" == 0 ]] && python -m experiments.completed "$summary_path" "$frames"; then
    echo "[reuse] $summary_path"
    return
  fi
  python -m "$module" --backend "$backend" --dtype "$run_dtype" --data-root "$data_root" \
    --split "$split" --frames "$frames" --warmup "$warmup" --timing-repeats "$repeats" \
    --timing-warmup-repeats "$timing_warmup" --sweeps "$sweeps" --device "$device" --log-dir "$log_dir"
}
run_case gtsparse.e2e_v2.kitti_second dataset/kitti test 1 second
run_case gtsparse.e2e_v2.nuscenes_voxelnext dataset/nuscenes test 1 voxelnext
run_case gtsparse.e2e_v2.nuscenes_voxelnext dataset/nuscenes test 10 voxelnext
run_case gtsparse.e2e_v2.semantickitti_sparse_resunet42 dataset/semantickitti val 1 minkunet
