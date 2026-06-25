# GPU Benchmarking & Telemetry Logger

`gpu-telemetry-benchmark` is a Python validation harness for running controlled AI/HPC-style PyTorch workloads while collecting GPU and system telemetry. It produces structured evidence files for each run: system inventory, benchmark results, telemetry CSV, machine-readable validation summary, and an engineering-style Markdown report.

The project is designed as a portfolio demonstration for AI hardware validation roles. It is intentionally honest about the system under test: it uses CUDA and `nvidia-smi` when an NVIDIA GPU is available, and it clearly labels CPU-only fallback runs as workflow checks rather than GPU hardware validation evidence.

## Relevance to AI Hardware Validation

AI hardware validation is not only about running a benchmark. A useful validation workflow defines the system under test, fixes workload parameters, records driver/runtime versions, captures telemetry during execution, evaluates pass/fail criteria, and documents anomalies with next actions.

This project demonstrates that workflow at a practical scale:

- Defines a reproducible test setup and records the hardware/software inventory.
- Runs controlled workloads that resemble common AI/HPC validation patterns.
- Captures GPU temperature, power, utilization, memory, clocks, PCIe data, and CPU/RAM telemetry where supported.
- Classifies runs as `PASS`, `PASS WITH WARNINGS`, or `FAIL` using explicit criteria.
- Separates clean GPU evidence from CPU fallback or telemetry-limited runs.

## What This Demonstrates

- **GPU telemetry:** Background `nvidia-smi` sampling for temperature, power, utilization, memory, clocks, PCIe link fields, and throttle indicators where available.
- **AI workload benchmarking:** Self-contained PyTorch `matmul`, `conv2d`, and lightweight transformer workloads with latency and throughput metrics.
- **Linux/system diagnostics:** System inventory collection for OS, kernel, Python, PyTorch, CUDA, cuDNN, CPU, RAM, disk, GPU inventory, and NVIDIA driver state.
- **Validation test planning:** Test matrix, issue report template, pass/fail criteria, configurable thresholds, and repeatable workload parameters.
- **Technical documentation:** README, validation report template, troubleshooting runbook, issue template, examples, and CV/interview preparation notes.
- **Reproducible reporting:** Per-run JSON/CSV/Markdown artifacts designed for review, comparison, and issue triage.

## Architecture

```text
CLI command
  |
  v
benchmark.py
  |-- system_info.py  -> system_info.json
  |-- workloads.py    -> PyTorch matmul, conv2d, transformer loops
  |-- telemetry.py    -> background nvidia-smi and psutil sampling
  |-- report.py       -> validation_summary.json and validation_report.md
  `-- utils.py        -> parsing, logging, stats, JSON helpers

outputs/<run_name>/
  |-- benchmark_results.json
  |-- system_info.json
  |-- telemetry.csv
  |-- validation_summary.json
  `-- validation_report.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If your platform needs a specific PyTorch wheel, install PyTorch from the official selector first, then run the editable install.

## Quickstart

CPU workflow check:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload matmul \
  --device auto \
  --duration-seconds 10 \
  --warmup-seconds 2 \
  --matrix-size 512 \
  --output-dir outputs/cpu_smoke
```

NVIDIA GPU run:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload conv2d \
  --device cuda \
  --gpu-index 0 \
  --duration-seconds 60 \
  --warmup-seconds 10 \
  --batch-size 32 \
  --output-dir outputs/gpu_conv2d
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

Longer stress-style run:

```bash
python -m gpu_telemetry_benchmark.benchmark \
  --workload matmul \
  --device cuda \
  --gpu-index 0 \
  --duration-seconds 900 \
  --warmup-seconds 30 \
  --matrix-size 4096 \
  --telemetry-interval 1 \
  --output-dir outputs/stress_matmul
```

## Workloads

| Workload | Purpose | Key Metrics |
| --- | --- | --- |
| `matmul` | Dense matrix multiplication for compute-heavy stress and approximate FLOP/s. | Iterations/sec, latency percentiles, approximate FLOP/s. |
| `conv2d` | CNN inference-style workload for batch-size and activation-memory behavior. | Samples/sec, latency percentiles, telemetry under varied batch sizes. |
| `transformer` | Lightweight transformer encoder workload for attention/MLP and sequence-length sensitivity. | Tokens/sec, sequences/sec, latency percentiles, CUDA memory metrics where available. |

All workloads are synthetic and self-contained. No large models are downloaded.

## Output Files

| File | Purpose |
| --- | --- |
| `system_info.json` | System under test inventory: OS, kernel, Python, PyTorch, CUDA, cuDNN, CPU, RAM, disk, GPU inventory, and NVIDIA driver data. |
| `telemetry.csv` | Time-series telemetry sampled during the benchmark. Uses `nvidia-smi` where available and `psutil` for CPU/RAM telemetry. |
| `benchmark_results.json` | Workload parameters, device selection, timing, throughput, latency statistics, memory metrics, status, errors, and notes. |
| `validation_summary.json` | Machine-readable classification, pass/fail criteria, anomalies, thresholds, telemetry summary, and artifact paths. |
| `validation_report.md` | Human-readable engineering report with scope, system details, results, telemetry summary, criteria, recommendations, limitations, and next steps. |

## Validation Criteria

Default criteria include:

- Device selection matched validation intent.
- Benchmark completed without runtime crash.
- No GPU out-of-memory error was recorded.
- Telemetry samples were captured.
- GPU telemetry was captured for CUDA runs.
- GPU temperature stayed below the configured threshold, default `85 C`.
- No thermal runaway trend was detected.
- Average GPU utilization met the configured threshold for CUDA runs, default `50%`.
- Required output files were created.

`PASS WITH WARNINGS` is expected for CPU fallback from `--device auto`, CUDA runs without GPU telemetry, or CUDA runs with low average utilization. Those runs may still be useful, but they are not clean validation baselines until the warnings are reviewed.

## Documentation

- [Validation test matrix](docs/test_matrix.md)
- [Validation report template](docs/validation_report_template.md)
- [Issue report template](docs/issue_report_template.md)
- [Linux/GPU troubleshooting runbook](docs/troubleshooting_runbook.md)
- [Synthetic example report](examples/example_validation_report.md)

## Limitations

- This project does not replace real enterprise GPU server validation.
- It does not validate liquid cooling loops, coolant flow, CDU behavior, facility water systems, or thermal-mechanical reliability.
- It does not validate GPU firmware, BIOS settings, BMC behavior, secure boot policy, or production driver qualification.
- It does not demonstrate rack-scale deployment, cluster scheduling, burn-in operations, or fleet monitoring experience.
- The workloads are synthetic and useful for validation workflow practice, not vendor-grade performance certification.
- CPU fallback runs verify the software/reporting workflow only; they do not validate GPU power, thermal, utilization, memory, or clock behavior.

## Next Improvements

- Add NVIDIA DCGM integration for enterprise-grade GPU health, diagnostics, and field coverage.
- Add NVML support as a lower-overhead telemetry path alongside `nvidia-smi`.
- Add multi-GPU scaling tests with explicit device placement and per-GPU artifact separation.
- Add NCCL communication benchmarks for all-reduce, bandwidth, latency, and topology-sensitive validation.
- Add thermal soak testing with longer-duration sampling, steady-state criteria, and trend analysis.
- Add Prometheus/Grafana integration for live dashboards and persistent telemetry storage.

## Tests

```bash
python -m pytest
```

The test suite is CPU-safe and does not require a GPU.
