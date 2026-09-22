#!/usr/bin/env bash
set -euo pipefail

gtsparse_root="$(cd "$(dirname "$0")" && pwd)"
cd "$gtsparse_root"
source scripts/activate.sh

run_end_to_end=0
run_microbenchmark=0
run_ablation=0
run_sensitivity=0
process_results=0
end_to_end_dtype=fp16
frames=0
micro_frames=100
memory_frames=20
warmup=20
timing_repeats=3
timing_warmup=2
device=cuda:0
output_root=.
overwrite=0
spconv_tile=observed

while [[ $# -gt 0 ]]; do
  case "$1" in
    --end-to-end)
      run_end_to_end=1
      if [[ "${2:-}" == "fp16" || "${2:-}" == "fp32" || "${2:-}" == "both" ]]; then
        end_to_end_dtype="$2"
        shift
      fi
      ;;
    --dtype)
      end_to_end_dtype="$2"
      shift
      ;;
    --microbenchmark) run_microbenchmark=1 ;;
    --ablation) run_ablation=1 ;;
    --sensitivity) run_sensitivity=1 ;;
    --plots) process_results=1 ;;
    --all)
      run_end_to_end=1
      run_microbenchmark=1
      run_ablation=1
      run_sensitivity=1
      ;;
    --frames)
      frames="$2"
      shift
      ;;
    --micro-frames)
      micro_frames="$2"
      shift
      ;;
    --memory-frames)
      memory_frames="$2"
      shift
      ;;
    --warmup)
      warmup="$2"
      shift
      ;;
    --timing-repeats)
      timing_repeats="$2"
      shift
      ;;
    --device)
      device="$2"
      shift
      ;;
    --output-root)
      output_root="$2"
      shift
      ;;
    --overwrite) overwrite=1 ;;
    --spconv-tile)
      spconv_tile="$2"
      shift
      ;;
    *)
      echo "usage: bash run_artifact.sh [--end-to-end [fp16|fp32|both]] [--microbenchmark] [--ablation] [--sensitivity] [--plots] [--all] [--frames N] [--micro-frames N] [--memory-frames N] [--warmup N] [--timing-repeats N] [--device DEVICE] [--output-root DIR] [--spconv-tile observed|autotuned] [--overwrite]"
      exit 2
      ;;
  esac
  shift
done

if [[ "$run_end_to_end" == 0 && "$run_microbenchmark" == 0 && "$run_ablation" == 0 && "$run_sensitivity" == 0 && "$process_results" == 0 ]]; then
  echo "usage: bash run_artifact.sh [--end-to-end [fp16|fp32|both]] [--microbenchmark] [--ablation] [--sensitivity] [--plots] [--all] [--spconv-tile observed|autotuned]"
  exit 2
fi

gpu_key=""
if [[ "$run_end_to_end" == 1 || "$run_microbenchmark" == 1 || "$run_ablation" == 1 || "$run_sensitivity" == 1 ]]; then
  gpu_key="$(python -c 'import torch; s=torch.cuda.get_device_name(torch.device("'"$device"'")); print("".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_"))')"
fi

run_experiment() {
  local marker="$1" expected_frames="$2"
  shift 2
  if [[ "$overwrite" == 0 ]] && python -m experiments.completed "$marker" "$expected_frames"; then
    echo "[reuse] $marker"
    return
  fi
  "$@"
}

if [[ "$run_end_to_end" == 1 ]]; then
  if [[ "$end_to_end_dtype" == "fp16" || "$end_to_end_dtype" == "both" ]]; then
    for backend in gtsparse spconv torchsparse minkowski; do
      bash run_e2e_v2.sh "$backend" fp16 "$frames" "$warmup" "$timing_repeats" "$timing_warmup" "$device" "$output_root/logs/end_to_end" "$overwrite"
    done
  fi
  if [[ "$end_to_end_dtype" == "fp32" || "$end_to_end_dtype" == "both" ]]; then
    for backend in gtsparse spconv torchsparse minkowski; do
      [[ "$end_to_end_dtype" == "both" && "$backend" == "minkowski" ]] && continue
      bash run_e2e_v2.sh "$backend" fp32 "$frames" "$warmup" "$timing_repeats" "$timing_warmup" "$device" "$output_root/logs/end_to_end" "$overwrite"
    done
  fi
fi

if [[ "$run_microbenchmark" == 1 ]]; then
  for backend in gtsparse spconv torchsparse minkowski; do
    run_dtype=fp16
    [[ "$backend" == "minkowski" ]] && run_dtype=fp32
    dtype_label=float16
    [[ "$run_dtype" == "fp32" ]] && dtype_label=float32
    marker="$output_root/logs/microbenchmark/timing/logs_${gpu_key}_${dtype_label}_voxelnext_nuscenes_sweeps1/${backend}.summary.json"
    run_experiment "$marker" "$micro_frames" python -m gtsparse.e2e_v2.nuscenes_voxelnext --backend "$backend" --dtype "$run_dtype" \
      --data-root dataset/nuscenes --split test --sweeps 1 --frames "$micro_frames" \
      --warmup "$warmup" --timing-repeats "$timing_repeats" --timing-warmup-repeats "$timing_warmup" \
      --device "$device" --log-dir "$output_root/logs/microbenchmark/timing"
  done

  for workload in second_kitti_sweeps1 voxelnext_nuscenes_sweeps1 voxelnext_nuscenes_sweeps10 minkunet_semantickitti_sweeps1; do
    case_dir="$output_root/logs/microbenchmark/profile/logs_${gpu_key}_float16_${workload}"
    run_experiment "${case_dir}/gtsparse.jsonl" "$micro_frames" python -m experiments.profile_workload --workload "$workload" --frames "$micro_frames" \
      --first-frames --warmup "$warmup" --device "$device" --out "${case_dir}/gtsparse.jsonl"
  done
  for workload in second_kitti_sweeps1 voxelnext_nuscenes_sweeps1 voxelnext_nuscenes_sweeps10 minkunet_semantickitti_sweeps1; do
    profile_frames="$frames"
    if [[ "$profile_frames" == 0 ]]; then
      case "$workload" in
        second_kitti_sweeps1) profile_frames=7518 ;;
        voxelnext_nuscenes_sweeps1|voxelnext_nuscenes_sweeps10) profile_frames=6008 ;;
        minkunet_semantickitti_sweeps1) profile_frames=4071 ;;
      esac
    fi
    case_dir="$output_root/logs/microbenchmark/template_profile/logs_${gpu_key}_float16_${workload}"
    run_experiment "${case_dir}/gtsparse.jsonl" "$profile_frames" python -m experiments.profile_templates --workload "$workload" --frames "$frames" \
      --warmup "$warmup" --device "$device" --out "${case_dir}/gtsparse.jsonl"
  done
  case_dir="$output_root/logs/microbenchmark/profile_spconv/logs_${gpu_key}_float16_voxelnext_nuscenes_sweeps1"
  run_experiment "${case_dir}/spconv.jsonl" "$micro_frames" python -m experiments.profile_spconv --workload voxelnext_nuscenes_sweeps1 --frames "$micro_frames" \
    --first-frames --warmup "$warmup" --device "$device" --out "${case_dir}/spconv.jsonl"

  for workload in voxelnext_nuscenes_sweeps1 voxelnext_nuscenes_sweeps10; do
    for backend in gtsparse spconv torchsparse minkowski; do
      run_dtype=fp16
      [[ "$backend" == "minkowski" ]] && run_dtype=fp32
      dtype_label=float16
      [[ "$run_dtype" == "fp32" ]] && dtype_label=float32
      case_dir="$output_root/logs/microbenchmark/time_breakdown/logs_${gpu_key}_${dtype_label}_${workload}"
      run_experiment "${case_dir}/${backend}.jsonl" "$micro_frames" python -m experiments.time_breakdown --workload "$workload" --backend "$backend" --dtype "$run_dtype" \
        --frames "$micro_frames" --warmup "$warmup" --device "$device" \
        --out "${case_dir}/${backend}.jsonl"
    done
  done

  for workload in second_kitti_sweeps1 voxelnext_nuscenes_sweeps1 voxelnext_nuscenes_sweeps10 minkunet_semantickitti_sweeps1; do
    for backend in gtsparse spconv torchsparse minkowski; do
      run_dtype=fp16
      [[ "$backend" == "minkowski" ]] && run_dtype=fp32
      dtype_label=float16
      [[ "$run_dtype" == "fp32" ]] && dtype_label=float32
      case_dir="$output_root/logs/microbenchmark/peak_memory/logs_${gpu_key}_${dtype_label}_${workload}"
      run_experiment "${case_dir}/${backend}.json" "$memory_frames" python -m experiments.peak_memory --workload "$workload" --backend "$backend" --dtype "$run_dtype" \
        --frames "$memory_frames" --warmup "$warmup" --device "$device" \
        --out "${case_dir}/${backend}.json"
    done
  done
fi

if [[ "$run_ablation" == 1 ]]; then
  for min_template in 0 1 4 7; do
    marker="$output_root/logs/ablation/min_template_${min_template}/logs_${gpu_key}_float16_voxelnext_nuscenes_sweeps10/gtsparse.summary.json"
    run_experiment "$marker" "$frames" env GTSPARSE_MIN_TEMPLATE="$min_template" python -m gtsparse.e2e_v2.nuscenes_voxelnext \
      --backend gtsparse --dtype fp16 --data-root dataset/nuscenes --split test --sweeps 10 \
      --frames "$frames" --warmup "$warmup" --timing-repeats "$timing_repeats" \
      --timing-warmup-repeats "$timing_warmup" --device "$device" \
      --log-dir "$output_root/logs/ablation/min_template_${min_template}"
  done
fi

if [[ "$run_sensitivity" == 1 ]]; then
  for sweeps in 1 5 10 20; do
    for backend in gtsparse spconv torchsparse minkowski; do
      run_dtype=fp16
      [[ "$backend" == "minkowski" ]] && run_dtype=fp32
      dtype_label=float16
      [[ "$run_dtype" == "fp32" ]] && dtype_label=float32
      marker="$output_root/logs/sensitivity/logs_${gpu_key}_${dtype_label}_voxelnext_nuscenes_sweeps${sweeps}/${backend}.summary.json"
      run_experiment "$marker" "$frames" python -m gtsparse.e2e_v2.nuscenes_voxelnext --backend "$backend" --dtype "$run_dtype" \
        --data-root dataset/nuscenes --split test --sweeps "$sweeps" --frames "$frames" \
        --warmup "$warmup" --timing-repeats "$timing_repeats" --timing-warmup-repeats "$timing_warmup" \
        --device "$device" --log-dir "$output_root/logs/sensitivity"
    done
  done
fi

python -m experiments.aggregate --logs "$output_root/logs" --results "$output_root/results" --spconv-tile "$spconv_tile"
python -m experiments.plot_results --results "$output_root/results" --figures "$output_root/figures"
