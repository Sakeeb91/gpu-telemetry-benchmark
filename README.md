# GPU Benchmarking & Telemetry Logger

`gpu-telemetry-benchmark` is a practical AI/HPC hardware validation mini-project. It runs controlled PyTorch workloads, collects GPU and system telemetry during execution, and produces structured evidence files that can be reviewed like a validation run: system inventory, telemetry CSV, benchmark results JSON, and a Markdown validation report.

The project is intentionally honest about the system under test. It works on NVIDIA GPU hosts when CUDA and `nvidia-smi` are available, and it falls back to CPU mode with clear reporting when no GPU is present. It does not claim H100, H200, B200, or data-center server validation unless the machine actually provides that hardware.

## Why It Matters

AI hardware validation is not just running a benchmark. A useful validation workflow defines the system under test, fixes workload parameters, records software and driver versions, captures telemetry, checks pass/fail criteria, and documents anomalies with next actions. This project demonstrates that workflow in a reproducible Python package.

## Architecture

```text
CLI command
  |
  v
benchmark.py
  |-- system_info.py  -> system_info.json
  |-- workloads.py    -> PyTorch matmul, conv2d, transformer loops
  |-- telemetry.py    -> background nvidia-smi and psutil sampling
  |-- utils.py        -> parsing, logging, stats, JSON helpers
  |
  v
outputs/<run_name>/
  |-- benchmark_results.json
  |-- telemetry.csv
  |-- system_info.json
  `-- validation_report.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If your platform needs a specific PyTorch wheel, install PyTorch from the official selector first, then run the editable install:

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

CPU-safe smoke run:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload matmul \
  --device cpu \
  --duration-seconds 10 \
  --warmup-seconds 2 \
  --matrix-size 512 \
  --output-dir outputs/cpu_smoke
```

Automatic device selection:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload conv2d \
  --device auto \
  --duration-seconds 60 \
  --warmup-seconds 10 \
  --batch-size 32 \
  --output-dir outputs/conv2d_run
```

## Example Commands

Matrix multiplication benchmark:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload matmul \
  --device auto \
  --duration-seconds 60 \
  --warmup-seconds 10 \
  --matrix-size 4096 \
  --output-dir outputs/matmul_run
```

CNN inference-style benchmark:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload conv2d \
  --device auto \
  --duration-seconds 60 \
  --warmup-seconds 10 \
  --batch-size 32 \
  --output-dir outputs/conv2d_run
```

Transformer-like workload:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload transformer \
  --device auto \
  --duration-seconds 60 \
  --warmup-seconds 10 \
  --batch-size 8 \
  --sequence-length 512 \
  --output-dir outputs/transformer_run
```

Batch-size sweep:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload conv2d \
  --device auto \
  --sweep-batch-sizes 1,2,4,8,16,32,64 \
  --duration-seconds 30 \
  --output-dir outputs/batch_sweep
```

Longer stress run:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload matmul \
  --device auto \
  --duration-seconds 900 \
  --warmup-seconds 30 \
  --matrix-size 4096 \
  --telemetry-interval 1 \
  --output-dir outputs/stress_test
```

## Output Files

Each single benchmark run creates:

| File | Purpose |
| --- | --- |
| `system_info.json` | System under test inventory: OS, Python, PyTorch, CUDA, CPU, RAM, GPU list, driver data, disk summary, hostname, timestamp. |
| `telemetry.csv` | Time-series telemetry sampled during the run. Uses `nvidia-smi` where available and `psutil` for CPU/RAM telemetry. |
| `benchmark_results.json` | Workload parameters, timing, latency statistics, throughput metrics, status, errors, and notes. |
| `validation_report.md` | Human-readable validation report with executive summary, telemetry summary, pass/fail criteria, anomalies, recommendations, limitations, and next steps. |

## CPU-Only Machines

Use `--device cpu` for explicit CPU runs, or `--device auto` to allow CPU fallback. CPU-only reports clearly state that GPU telemetry was unavailable or telemetry-limited.

## NVIDIA GPU Machines

Use `--device auto` or `--device cuda`. `--device cuda` fails clearly if CUDA is unavailable. GPU telemetry requires `nvidia-smi`; if `nvidia-smi` is missing, the benchmark still runs and records CPU/RAM telemetry.

## Interpreting the Validation Report

The generated report classifies each run as:

| Classification | Meaning |
| --- | --- |
| `PASS` | Benchmark completed and required criteria were met. |
| `PASS WITH WARNINGS` | Benchmark completed, but one or more non-critical validation criteria need review. |
| `FAIL` | A runtime error, CUDA out-of-memory event, missing required artifact, or threshold violation was detected. |

Default criteria include completion status, telemetry file creation, system inventory creation, GPU temperature below 85 C, and average GPU utilization above 50 percent for GPU runs.

## Limitations

- Workloads are synthetic and are designed for validation workflow practice, not vendor-grade performance certification.
- Telemetry availability depends on host tools, driver support, and `nvidia-smi` query support.
- CPU fallback is useful for development and reproducibility, but it cannot validate GPU thermal, power, or utilization behavior.
- No large models are downloaded. The transformer workload is intentionally lightweight and self-contained.

## Future Improvements

- Add repeat-run variance analysis and regression thresholds.
- Add CSV/HTML summary dashboards for batch sweeps.
- Add optional NVML support for lower-overhead telemetry.
- Add multi-GPU workload placement and topology-aware tests.
- Add container recipes for fully pinned runtime environments.

## CV Bullets

- Built a Python GPU validation harness that runs PyTorch AI/HPC workloads while collecting synchronized GPU, CPU, memory, power, and thermal telemetry.
- Implemented reproducible benchmark reporting with system inventory, structured JSON/CSV outputs, pass/fail criteria, anomaly detection, and Markdown validation reports.
- Designed CPU-safe fallback behavior and graceful handling for missing CUDA, missing `nvidia-smi`, runtime errors, and CUDA out-of-memory conditions.
- Documented validation test matrices, issue-report templates, and Linux/GPU troubleshooting runbooks for AI hardware diagnostics.
