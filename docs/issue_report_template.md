# Hardware/Software Issue Report Template

## Issue Title

Short, specific title.

## System Under Test

- Hostname:
- OS/kernel:
- CPU:
- RAM:
- GPU model(s):
- NVIDIA driver:
- CUDA/PyTorch versions:

## Environment

- Repository commit:
- Command run:
- Workload:
- Parameters:
- Output directory:

## Steps To Reproduce

1. 
2. 
3. 

## Expected Behavior

Describe the expected benchmark or telemetry behavior.

## Actual Behavior

Describe what happened, including failure messages.

## Telemetry Observed

- Peak GPU temperature:
- Average GPU utilization:
- Peak GPU memory used:
- Average power draw:
- CPU/RAM behavior:
- Clock or throttle observations:

## Logs

Paste relevant `benchmark_results.json` fields, terminal output, or system logs.

## Suspected Cause

Hardware, driver, thermal, power, workload parameter, software dependency, or unknown.

## Severity

- `Critical`: Blocks validation or risks hardware stability.
- `High`: Benchmark fails or produces unusable evidence.
- `Medium`: Results complete but telemetry or metrics are degraded.
- `Low`: Documentation, usability, or non-blocking issue.

## Recommended Next Action

State the next diagnostic or remediation step.
