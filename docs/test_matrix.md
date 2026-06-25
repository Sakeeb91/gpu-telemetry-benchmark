# Test Matrix

| Test ID | Workload | Parameter Varied | Expected Behavior | Metrics Captured | Pass/Fail Criteria |
| --- | --- | --- | --- | --- | --- |
| TM-001 | `matmul` | `matrix-size` | Dense GEMM loop completes for configured duration. | Iterations/sec, average latency, p50, p95, p99, approximate FLOP/s, telemetry. | Run completes, no OOM, telemetry and system inventory created, GPU temperature below threshold. |
| TM-002 | `conv2d` | `batch-size` | CNN inference loop scales throughput as batch size changes, subject to memory limits. | Samples/sec, batch latency stats, GPU utilization, memory used, power draw. | Run completes, no OOM, GPU utilization acceptable for GPU runs, no thermal threshold violation. |
| TM-003 | `transformer` | `sequence-length` | Lightweight transformer encoder completes without downloading models. | Tokens/sec, sequences/sec, latency stats, peak memory when available. | Run completes, no OOM, telemetry artifacts created. |
| TM-004 | `conv2d` | Sweep batch sizes `1..64` | Produces one sub-run per batch size and an aggregate sweep summary. | Throughput by batch size, latency stats, telemetry per sub-run. | Each sub-run completes or records a clear failure reason. |
| TM-005 | `matmul` | Duration and telemetry interval | Sustained workload records stable telemetry for stress duration. | Temperature trend, power draw, clocks, utilization, memory, CPU/RAM. | No crash, no thermal runaway, no missing telemetry samples. |
| TM-006 | Any | `--device cuda` when CUDA unavailable | CLI exits clearly and records failure when possible. | Error message, system inventory, failed result JSON if output dir is available. | No silent failure or misleading CPU fallback. |
| TM-007 | Any | `--device auto` on CPU-only host | CPU fallback runs and report states GPU telemetry is unavailable. | CPU/RAM telemetry, workload metrics. | Run completes and report does not claim GPU validation. |
| TM-008 | Any | Missing `nvidia-smi` | Benchmark continues with system telemetry only. | CPU/RAM telemetry, warning in results/report. | No crash due to missing NVIDIA telemetry utility. |
