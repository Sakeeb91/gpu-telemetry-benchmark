# Validation Report: matmul

This is a synthetic sample report for documentation only. It demonstrates the expected report style from a CPU fallback run and does not claim NVIDIA GPU validation.

## Executive Summary

- Classification: **PASS WITH WARNINGS**
- Workload: `matmul`
- Device used: `cpu`
- Status: `completed`
- Duration: 1.00 s
- Primary throughput: 465.00 matmuls/sec
- Engineering assessment: Run is useful for workflow verification only; it is not GPU hardware validation evidence.

## Validation Scope

- Validation focus: Dense GEMM throughput, tensor-core or SIMD utilization, memory allocation stability.
- Resource profile: Compute-heavy with predictable matrix memory footprint.
- Evidence scope: CPU-only software workflow validation; no GPU hardware behavior validated.
- Random seed: 1234
- GPU index selected: N/A

## Results Summary

- Total iterations: 465
- Average latency: 2.15 ms
- p95 latency: 2.88 ms
- Error: None

## Telemetry Summary

- Telemetry samples: 2
- GPU telemetry samples: 0
- Peak GPU temperature: N/A
- Average GPU utilization: N/A
- Average CPU utilization: 15.30%

## Detected Anomalies

- Device selection matched validation intent: GPU unavailable; CPU fallback used. This is not GPU validation evidence.

## Pass/Fail Criteria

| Criterion | Status | Evidence |
| --- | --- | --- |
| Device selection matched validation intent | WARN | GPU unavailable; CPU fallback used. This is not GPU validation evidence. |
| Benchmark completed without runtime crash | PASS | status=completed |
| Telemetry samples were captured | PASS | 2 sample(s) |

## Limitations

- CPU mode cannot validate GPU power, thermal, clock, memory, or utilization behavior.
- Synthetic sample output is for repository documentation only.
