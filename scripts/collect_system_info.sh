#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:-outputs/system_info}"

python -m gpu_telemetry_benchmark.system_info --output-dir "${OUTPUT_DIR}"
