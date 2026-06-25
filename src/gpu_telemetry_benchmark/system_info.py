"""System inventory collection for validation runs."""

from __future__ import annotations

import argparse
import platform
import re
import socket
import sys
from pathlib import Path
from typing import Any

from .utils import bytes_to_gb, run_command, utc_now_iso, write_json


def _optional_psutil() -> Any | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    return psutil


def _cpu_model() -> str | None:
    system = platform.system()
    if system == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    if system == "Darwin":
        code, stdout, _ = run_command(["sysctl", "-n", "machdep.cpu.brand_string"], 3)
        if code == 0 and stdout:
            return stdout.strip()
    processor = platform.processor()
    return processor or None


def _parse_nvidia_cuda_version(nvidia_smi_output: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_output)
    if match:
        return match.group(1)
    return None


def _nvidia_driver_version() -> str | None:
    code, stdout, _ = run_command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
        5,
    )
    if code != 0 or not stdout:
        return None
    return stdout.splitlines()[0].strip()


def _nvidia_smi_l() -> str | None:
    code, stdout, _ = run_command(["nvidia-smi", "-L"], 5)
    if code != 0:
        return None
    return stdout


def _nvidia_gpu_inventory() -> list[dict[str, str]]:
    fields = [
        "index",
        "name",
        "uuid",
        "pci.bus_id",
        "driver_version",
        "memory.total",
        "power.limit",
    ]
    code, stdout, _ = run_command(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
        5,
    )
    if code != 0 or not stdout:
        return []
    import csv

    inventory: list[dict[str, str]] = []
    output_names = [
        "index",
        "name",
        "uuid",
        "pci_bus_id",
        "driver_version",
        "memory_total_mb",
        "power_limit_w",
    ]
    for row in csv.reader(stdout.splitlines()):
        inventory.append({name: value.strip() for name, value in zip(output_names, row, strict=False)})
    return inventory


def _nvidia_cuda_version() -> str | None:
    code, stdout, _ = run_command(["nvidia-smi"], 5)
    if code != 0 or not stdout:
        return None
    return _parse_nvidia_cuda_version(stdout)


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "pytorch_installed": False,
        "pytorch_version": None,
        "cuda_available_from_pytorch": False,
        "pytorch_cuda_version": None,
        "cudnn_version": None,
        "gpu_count": 0,
        "gpu_names_detected_by_pytorch": [],
    }
    try:
        import torch
    except ImportError as exc:
        info["pytorch_import_error"] = str(exc)
        return info

    info["pytorch_installed"] = True
    info["pytorch_version"] = torch.__version__
    info["cuda_available_from_pytorch"] = bool(torch.cuda.is_available())
    info["pytorch_cuda_version"] = torch.version.cuda
    try:
        info["cudnn_version"] = torch.backends.cudnn.version()
    except Exception as exc:  # pragma: no cover - backend-specific
        info["cudnn_version_error"] = str(exc)
    if torch.cuda.is_available():
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names_detected_by_pytorch"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    return info


def collect_system_info() -> dict[str, Any]:
    """Collect host, runtime, and accelerator inventory for a validation report."""
    psutil = _optional_psutil()
    virtual_memory = psutil.virtual_memory() if psutil else None
    disk_usage = psutil.disk_usage("/") if psutil else None

    info: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "hostname": socket.gethostname(),
        "os_name": platform.system(),
        "os_version": platform.version(),
        "platform": platform.platform(),
        "kernel_version": platform.release(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu_model": _cpu_model(),
        "cpu_core_count_logical": psutil.cpu_count(logical=True) if psutil else None,
        "cpu_core_count_physical": psutil.cpu_count(logical=False) if psutil else None,
        "ram_total_gb": bytes_to_gb(virtual_memory.total) if virtual_memory else None,
        "disk_root_total_gb": bytes_to_gb(disk_usage.total) if disk_usage else None,
        "disk_root_used_gb": bytes_to_gb(disk_usage.used) if disk_usage else None,
        "disk_root_free_gb": bytes_to_gb(disk_usage.free) if disk_usage else None,
        "disk_root_percent": disk_usage.percent if disk_usage else None,
        "nvidia_driver_version": _nvidia_driver_version(),
        "nvidia_cuda_version_from_smi": _nvidia_cuda_version(),
        "nvidia_smi_l": _nvidia_smi_l(),
        "nvidia_gpu_inventory": _nvidia_gpu_inventory(),
    }
    info.update(_torch_info())
    return info


def main(argv: list[str] | None = None) -> int:
    """Write system inventory to an output directory."""
    parser = argparse.ArgumentParser(description="Collect system inventory for a validation run.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/system_info"))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "system_info.json"
    write_json(output_path, collect_system_info())
    print(output_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
