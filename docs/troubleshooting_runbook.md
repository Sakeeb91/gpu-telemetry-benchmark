# Linux/GPU Troubleshooting Runbook

Use these commands to inspect the system under test before or after a benchmark run.

| Command | What It Is For |
| --- | --- |
| `nvidia-smi` | Shows GPU inventory, driver version, CUDA runtime version reported by the driver, memory use, utilization, temperature, and active processes. |
| `nvidia-smi -L` | Lists detected NVIDIA GPUs and UUIDs. Useful for confirming GPU count and model names. |
| `nvidia-smi topo -m` | Shows GPU, CPU, and interconnect topology. Useful for multi-GPU placement and PCIe/NVLink awareness. |
| `lspci | grep -i nvidia` | Confirms NVIDIA PCIe devices are visible to the OS. |
| `lscpu` | Shows CPU model, socket/core/thread layout, NUMA nodes, and supported instruction sets. |
| `free -h` | Shows system memory and swap usage in human-readable units. |
| `df -h` | Shows filesystem capacity and free space. Useful when logs or datasets fill a disk. |
| `lsblk` | Shows block devices, partitions, and mount points. Useful for identifying local disks. |
| `dmesg -T | grep -i -E "nvrm|xid|cuda|pcie|error"` | Searches kernel logs for NVIDIA, CUDA, PCIe, or hardware errors. Requires appropriate permissions on some systems. |
| `journalctl -k -b` | Shows kernel logs from the current boot on systemd systems. Useful when `dmesg` is restricted. |
| `uname -a` | Shows kernel and OS build details. |
| `sensors` | Shows temperature and voltage sensors when `lm-sensors` is installed and configured. Useful for CPU/platform thermal checks. |

## Suggested Triage Flow

1. Confirm the OS sees the GPU with `nvidia-smi -L`.
2. Confirm the driver is healthy with `nvidia-smi`.
3. Confirm CPU, RAM, and disk state with `lscpu`, `free -h`, and `df -h`.
4. Run a short CPU benchmark to confirm the Python environment works.
5. Run a short GPU benchmark with conservative parameters.
6. If the run fails, inspect `benchmark_results.json`, `validation_report.md`, `dmesg`, and `journalctl`.
7. If temperatures, clocks, or power look abnormal, repeat with a shorter duration and lower workload size before attempting a stress test.
