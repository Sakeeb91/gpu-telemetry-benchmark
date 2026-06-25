# Validation Report Template

## Executive Summary

- Classification:
- Workload:
- Device requested:
- Device used:
- Status:
- Duration:
- Primary throughput:
- Engineering assessment:
- Notable warnings or failures:

## Validation Scope

- Validation focus:
- Resource profile:
- Evidence scope:
- Random seed:
- GPU index selected:
- What this run can prove:
- What this run cannot prove:

## System Under Test

- Hostname:
- Server/workstation model:
- OS:
- Kernel:
- CPU:
- CPU socket/core/thread layout:
- RAM:
- GPU count:
- GPU names:
- GPU UUIDs / PCI bus IDs:
- NVIDIA driver:
- Power/cooling notes:

## Software Stack

- Python:
- PyTorch:
- CUDA available from PyTorch:
- PyTorch CUDA version:
- cuDNN version:
- CUDA version from `nvidia-smi`:
- Virtualenv/container:

## Benchmark Workload

- Workload name:
- Parameters:
- Warmup duration:
- Measured duration:
- Command:

## Results Summary

- Total iterations:
- Units processed:
- Throughput:
- Average latency:
- p50 latency:
- p95 latency:
- p99 latency:
- Status:
- Errors:
- CUDA memory metrics:

## Telemetry Summary

- Telemetry source:
- Telemetry sample count:
- GPU telemetry sample count:
- First sample:
- Last sample:
- Peak GPU temperature:
- GPU temperature delta:
- Average GPU utilization:
- Peak GPU memory usage:
- Average power draw:
- Clock/throttle observations:
- CPU/RAM observations:

## Thresholds

- GPU temperature threshold:
- Minimum average GPU utilization for CUDA runs:
- Notes about workload-specific interpretation:

## Detected Anomalies

- 

## Pass/Fail Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Device selection matched validation intent |  |  |
| Benchmark completed without runtime crash |  |  |
| No GPU out-of-memory error |  |  |
| Telemetry samples were captured |  |  |
| GPU telemetry was captured for CUDA runs |  |  |
| GPU temperature stayed below threshold |  |  |
| No thermal runaway trend detected |  |  |
| Average GPU utilization exceeded threshold for GPU runs |  |  |
| Telemetry file was created |  |  |
| System info file was created |  |  |

## Evidence Artifacts

- `system_info.json`:
- `benchmark_results.json`:
- `telemetry.csv`:
- `validation_summary.json`:
- `validation_report.md`:

## Recommendations

- 

## Limitations

- 

## Next Steps

- 
