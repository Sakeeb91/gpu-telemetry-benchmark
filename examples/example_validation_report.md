# Validation Report: matmul

This is a synthetic sample report for documentation only. It demonstrates the expected report style from a CPU fallback run and does not claim NVIDIA GPU validation.

## Executive Summary

- Classification: **PASS WITH WARNINGS**
- Workload: `matmul`
- Device used: `cpu`
- Status: `completed`
- Primary throughput: 465.00 matmuls/sec

## System Under Test

- Hostname: `sample-host`
- OS: Generic Linux sample
- CPU: Generic CPU
- GPU count from PyTorch: 0
- NVIDIA driver: N/A

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

- GPU unavailable; CPU fallback used.
- GPU telemetry was unavailable or telemetry-limited for this run.

## Limitations

- CPU mode cannot validate GPU power, thermal, clock, memory, or utilization behavior.
- Synthetic sample output is for repository documentation only.
