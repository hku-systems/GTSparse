# Example runs

Each experiment category has an example run that completes in a few minutes
on any GPU. The examples use the first N frames of the datasets you
downloaded; run them from the repository root with the environment activated.
The `--output-root examples` flag keeps all outputs under this directory, so
the main `logs/` directory is left untouched.

```bash
source scripts/activate.sh
bash run_artifact.sh --end-to-end fp16 --frames 5 --output-root examples
bash run_artifact.sh --end-to-end fp32 --frames 5 --output-root examples
bash run_artifact.sh --microbenchmark --micro-frames 10 --memory-frames 5 --frames 50 --output-root examples
bash run_artifact.sh --ablation --frames 50 --output-root examples
bash run_artifact.sh --sensitivity --frames 50 --output-root examples
```

`logs/` in this directory contains the expected outputs of exactly these
commands, captured on the reference RTX 3080. Compare your outputs against
them by structure and magnitude; your per-GPU directory names carry your
GPU's name, and absolute values differ across GPUs and clock settings:

- same experiment subdirectories and the same per-backend
  summary/config/jsonl files,
- end-to-end: GTSparse fastest on every FP16 workload,
- microbenchmark: average assigned template width below 27, GTSparse raw
  throughput above TS++ with a higher useful share, peak memory within the
  same order of magnitude,
- ablation: median latency rising as narrower template families are removed,
  except the center family,
- sensitivity: GTSparse's speedup over TS++ narrowing as sweeps grow.

The orderings and shares above are what the paper's claims are stated in.
