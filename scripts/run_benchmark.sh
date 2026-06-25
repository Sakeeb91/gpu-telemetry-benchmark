#!/usr/bin/env bash
set -euo pipefail

WORKLOAD="${1:-matmul}"
OUTPUT_DIR="${2:-outputs/${WORKLOAD}_smoke}"

python -m gpu_telemetry_benchmark.benchmark \
  --workload "${WORKLOAD}" \
  --device auto \
  --duration-seconds 10 \
  --warmup-seconds 2 \
  --matrix-size 512 \
  --matmul-dtype float32 \
  --batch-size 4 \
  --sequence-length 64 \
  --hidden-dim 64 \
  --num-heads 4 \
  --num-layers 1 \
  --seed 1234 \
  --output-dir "${OUTPUT_DIR}"
