# Linux/GPU Troubleshooting Runbook

Use these commands to inspect the system under test before or after a benchmark run. Some commands require Linux, root privileges, or NVIDIA drivers. Record command output in the issue report when a run fails or produces warnings.

## NVIDIA GPU And Driver

| Command | What It Is For |
| --- | --- |
| `nvidia-smi` | First-pass GPU health check: driver version, CUDA runtime reported by the driver, memory use, utilization, temperature, power, and active processes. |
| `watch -n 1 nvidia-smi` | Live view during a run to confirm the workload lands on the expected GPU. |
| `nvidia-smi -L` | Lists detected NVIDIA GPUs and UUIDs. Useful for confirming GPU count and model names. |
| `nvidia-smi topo -m` | Shows GPU, CPU, PCIe, and NVLink topology. Useful for multi-GPU placement and NUMA awareness. |
| `nvidia-smi -q` | Detailed GPU state, including clocks, power, temperature, ECC, retired pages, and firmware details where supported. |
| `nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER,CLOCK` | Focused query for throttling, thermal, power, and clock behavior. |
| `nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,temperature.gpu,power.draw,power.limit,utilization.gpu,memory.used,memory.total --format=csv` | Structured one-shot telemetry query matching the project’s validation evidence model. |
| `nvidia-smi dmon -s pucmt -d 1` | Lightweight live monitoring for power, utilization, clocks, memory, and temperature. |
| `nvidia-smi pmon -c 1` | Shows per-process GPU usage when supported. Helpful for checking whether another process polluted the run. |
| `systemctl status nvidia-persistenced` | Confirms NVIDIA persistence daemon state on Linux systems that use it. |
| `lsmod | grep nvidia` | Confirms NVIDIA kernel modules are loaded. |
| `modinfo nvidia | head` | Shows installed NVIDIA kernel module metadata. |
| `nvcc --version` | Shows CUDA toolkit compiler version if the toolkit is installed. Not required for PyTorch wheels. |

## PCIe, NUMA, And Platform Inventory

| Command | What It Is For |
| --- | --- |
| `lspci | grep -i -E "nvidia|3d|vga"` | Confirms GPU PCIe devices are visible to the OS. |
| `lspci -vv -s <PCI_BUS_ID>` | Shows PCIe link speed, width, ACS, and device details for a specific GPU. |
| `lscpu` | Shows CPU model, socket/core/thread layout, NUMA nodes, and supported instruction sets. |
| `numactl --hardware` | Shows NUMA topology and memory locality. Useful when interpreting multi-GPU or CPU-bound results. |
| `lstopo-no-graphics` | Shows hardware topology if `hwloc` is installed. Useful for CPU/GPU/NIC locality. |
| `dmidecode -t system -t baseboard -t bios` | Shows server, board, and BIOS details. Usually requires root. |
| `ipmitool sdr` | Shows BMC sensor readings such as inlet temperature, fan speed, and power where available. |
| `sensors` | Shows CPU/platform temperatures and voltages when `lm-sensors` is installed and configured. |

## OS, Kernel, And Error Logs

| Command | What It Is For |
| --- | --- |
| `uname -a` | Shows kernel and OS build details. |
| `cat /etc/os-release` | Shows Linux distribution and release details. |
| `dmesg -T | grep -i -E "nvrm|xid|cuda|pcie|aer|error|fault"` | Searches kernel logs for NVIDIA, CUDA, PCIe AER, and hardware errors. Requires permissions on some systems. |
| `journalctl -k -b` | Shows kernel logs from the current boot on systemd systems. Useful when `dmesg` is restricted. |
| `journalctl -u nvidia-persistenced -b` | Shows NVIDIA persistence daemon logs. |
| `mokutil --sb-state` | Checks Secure Boot state, which can affect driver module loading. |

## Runtime And Python Environment

| Command | What It Is For |
| --- | --- |
| `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"` | Confirms PyTorch import, CUDA visibility, and PyTorch CUDA build. |
| `python -m gpu_telemetry_benchmark.system_info --output-dir outputs/system_info_check` | Captures the project’s normalized system inventory. |
| `which python && python --version` | Confirms the interpreter used for the run. |
| `python -m pip freeze` | Captures installed Python packages for reproducibility. |
| `ldconfig -p | grep -i cuda` | Shows CUDA-related shared libraries visible to the linker on Linux. |

## Storage, Memory, And General Host Health

| Command | What It Is For |
| --- | --- |
| `free -h` | Shows system memory and swap usage in human-readable units. |
| `df -h` | Shows filesystem capacity and free space. Useful when logs or datasets fill a disk. |
| `lsblk` | Shows block devices, partitions, and mount points. |
| `iostat -xz 1 5` | Shows block-device utilization and latency if `sysstat` is installed. |
| `top` or `htop` | Shows CPU, memory, and process-level load during a run. |
| `uptime` | Shows load average and host uptime. |

## Suggested Triage Flow

1. Confirm the OS sees the GPU with `nvidia-smi -L`.
2. Confirm the driver is healthy with `nvidia-smi` and `lsmod | grep nvidia`.
3. Confirm PyTorch can see CUDA with `python -c "import torch; print(torch.cuda.is_available())"`.
4. Confirm CPU, RAM, disk, and topology with `lscpu`, `numactl --hardware`, `free -h`, and `df -h`.
5. Run a short CPU benchmark to confirm the Python environment and reporting pipeline work.
6. Run a short GPU benchmark with conservative parameters and `watch -n 1 nvidia-smi` in another terminal.
7. If the run fails, inspect `benchmark_results.json`, `validation_summary.json`, `validation_report.md`, `dmesg`, and `journalctl`.
8. If temperatures, clocks, or power look abnormal, collect `nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER,CLOCK`.
9. If the issue is multi-GPU specific, inspect `nvidia-smi topo -m`, `lspci -vv`, and NUMA topology.
10. Do not use a warning-limited run as a clean baseline until the warning is explained or accepted.
