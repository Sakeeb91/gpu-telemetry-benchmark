# Validation Test Matrix

This matrix defines repeatable validation scenarios for the benchmark harness. It is not a certification plan for any specific GPU server. Treat CPU-only runs as software workflow checks and CUDA runs with telemetry as GPU evidence.

| Test ID | Scenario | Workload | Parameter Varied | Preconditions | Evidence Artifacts | Metrics Captured | Acceptance Criteria |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TM-001 | CPU software smoke test | `matmul` | Small `matrix-size` | Any Python host with PyTorch installed | `system_info.json`, `benchmark_results.json`, `telemetry.csv`, `validation_summary.json`, `validation_report.md` | Iterations/sec, latency percentiles, CPU/RAM telemetry | Run completes, telemetry samples captured, report states CPU-only scope. |
| TM-002 | CUDA device selection | Any | `--device cuda`, `--gpu-index` | NVIDIA GPU host with CUDA-enabled PyTorch | Same as above | Device label, GPU index, workload metrics, GPU telemetry | Requested CUDA device is selected; invalid GPU index fails clearly. |
| TM-003 | Dense compute stress | `matmul` | `matrix-size`, `matmul-dtype`, duration | NVIDIA GPU host preferred | Same as above | Iterations/sec, average/p50/p95/p99 latency, approximate FLOP/s, temperature, power, clocks, utilization | Run completes, no OOM, GPU temperature below threshold, no thermal runaway trend, telemetry samples present. |
| TM-004 | CNN batch scaling | `conv2d` | `batch-size` or `--sweep-batch-sizes` | CPU or CUDA host; CUDA preferred for GPU evidence | Per-run artifacts plus `sweep_results.json` for sweeps | Samples/sec, latency percentiles, memory used, utilization, power | Each batch-size run completes or records a clear failure reason; throughput/latency trend is reviewable. |
| TM-005 | Transformer sequence sensitivity | `transformer` | `sequence-length`, `hidden-dim`, `batch-size` | CPU or CUDA host; CUDA preferred for GPU evidence | Same as above | Tokens/sec, sequences/sec, latency percentiles, memory metrics when CUDA is available | Run completes without OOM; report explains workload scope and any telemetry limitations. |
| TM-006 | Missing CUDA guardrail | Any | `--device cuda` on CPU-only host | Host without CUDA-enabled PyTorch | `benchmark_results.json`, `validation_report.md` if output dir can be created | Error type/message, system inventory | Tool fails clearly and does not silently fall back to CPU. |
| TM-007 | Auto CPU fallback | Any | `--device auto` on CPU-only host | Host without CUDA-enabled PyTorch | Same as above | CPU/RAM telemetry, workload metrics, warning classification | Run completes as `PASS WITH WARNINGS`; report states it is not GPU validation evidence. |
| TM-008 | Missing `nvidia-smi` | Any CUDA or CPU run | Host PATH/tool availability | Host where `nvidia-smi` is absent or inaccessible | Same as above | CPU/RAM telemetry, telemetry warning | Benchmark continues; report records telemetry limitation. CUDA run is warning-limited, not a clean GPU baseline. |
| TM-009 | Thermal guardrail | `matmul` | Long duration, telemetry interval | NVIDIA GPU host with `nvidia-smi` | Telemetry CSV and validation summary | Peak temperature, temperature delta, power draw, clocks | Temperature remains below configured threshold and no thermal runaway trend is detected. |
| TM-010 | OOM handling | `matmul`, `conv2d`, or `transformer` | Oversized matrix/batch/sequence | CUDA host or constrained CPU host | Results JSON and report | Error type/message, partial telemetry | OOM is captured in results/report; process does not fail silently or claim pass. |

## Suggested Baseline Runs

1. CPU smoke: `matmul`, 10 seconds, small matrix.
2. GPU smoke: `conv2d`, 30 seconds, batch size 8 or 16.
3. GPU stress: `matmul`, 15 minutes, telemetry interval 1 second.
4. Batch sweep: `conv2d`, batch sizes `1,2,4,8,16,32,64`.
5. Transformer memory sensitivity: sequence lengths `128,256,512`.

## Evidence Review Checklist

- Confirm `system_info.json` identifies the actual hardware and software stack.
- Confirm `benchmark_results.json` records parameters, seed, device, status, errors, and throughput.
- Confirm `telemetry.csv` has nonzero samples and GPU rows for CUDA validation runs.
- Confirm `validation_summary.json` classification matches the report.
- Confirm `validation_report.md` separates clean passes from warnings and failures.
