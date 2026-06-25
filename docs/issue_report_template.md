# Hardware/Software Issue Report Template

Use this template when a benchmark run exposes a failure, warning, suspicious telemetry trend, or reproducibility gap.

## Issue Title

Short, specific title. Example: `CUDA OOM during transformer sequence-length 1024 run on GPU 0`.

## System Under Test

- Hostname:
- Server/workstation model:
- OS/kernel:
- CPU:
- RAM:
- GPU model(s):
- GPU count:
- GPU index/UUID affected:
- NVIDIA driver:
- CUDA runtime from `nvidia-smi`:
- PyTorch version/CUDA version:
- Power/cooling configuration if known:

## Environment

- Repository commit:
- Command run:
- Workload:
- Parameters:
- Random seed:
- Output directory:
- Container/virtualenv details:
- User privileges:

## Steps To Reproduce

1. 
2. 
3. 

## Expected Behavior

Describe the expected benchmark, telemetry, and report behavior.

## Actual Behavior

Describe what happened, including failure messages and whether the run produced all artifacts.

## Validation Classification

- Report classification:
- Failed criteria:
- Warning criteria:
- Relevant entry from `validation_summary.json`:

## Telemetry Observed

- Telemetry sample count:
- GPU telemetry sample count:
- Peak GPU temperature:
- GPU temperature delta:
- Average GPU utilization:
- Peak GPU memory used:
- Average power draw:
- Clock behavior:
- Throttle reasons:
- PCIe link generation/width:
- CPU/RAM behavior:

## Logs And Command Output

Paste or attach relevant excerpts:

- `benchmark_results.json`
- `validation_summary.json`
- `validation_report.md`
- Terminal output
- `nvidia-smi`
- `nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER,CLOCK`
- `dmesg -T | grep -i -E "nvrm|xid|cuda|pcie|error"`
- `journalctl -k -b`

## Suspected Cause

Choose the most likely category and explain briefly:

- Hardware fault
- Driver/CUDA/PyTorch mismatch
- Thermal or airflow constraint
- Power limit or PSU/platform issue
- PCIe/topology issue
- Workload parameter too large
- Telemetry/tooling limitation
- Unknown

## Severity

- `Critical`: Hardware instability, repeated Xid errors, system crash, or validation blocked.
- `High`: Benchmark fails or produces unusable evidence.
- `Medium`: Run completes but telemetry, performance, or report criteria show suspicious behavior.
- `Low`: Documentation, usability, or non-blocking issue.

## Impact

- Affected workload(s):
- Affected GPU(s):
- Reproducibility: always/intermittent/once
- Blocks publication or baseline use: yes/no

## Recommended Next Action

State the next diagnostic or remediation step, owner, and target date.
