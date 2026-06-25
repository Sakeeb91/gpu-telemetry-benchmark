"""Background GPU and system telemetry collection."""

from __future__ import annotations

import csv
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any

from .utils import run_command, utc_now_iso


TELEMETRY_FIELDS = [
    "timestamp",
    "gpu_index",
    "gpu_name",
    "driver_version",
    "cuda_version",
    "temperature_gpu_c",
    "power_draw_w",
    "power_limit_w",
    "utilization_gpu_percent",
    "utilization_memory_percent",
    "memory_total_mb",
    "memory_used_mb",
    "memory_free_mb",
    "clocks_graphics_mhz",
    "clocks_memory_mhz",
    "pcie_link_gen_current",
    "pcie_link_width_current",
    "throttle_reasons",
    "cpu_percent",
    "ram_total_gb",
    "ram_used_gb",
    "ram_percent",
]


FULL_QUERY_FIELDS = [
    ("index", "gpu_index"),
    ("name", "gpu_name"),
    ("driver_version", "driver_version"),
    ("temperature.gpu", "temperature_gpu_c"),
    ("power.draw", "power_draw_w"),
    ("power.limit", "power_limit_w"),
    ("utilization.gpu", "utilization_gpu_percent"),
    ("utilization.memory", "utilization_memory_percent"),
    ("memory.total", "memory_total_mb"),
    ("memory.used", "memory_used_mb"),
    ("memory.free", "memory_free_mb"),
    ("clocks.gr", "clocks_graphics_mhz"),
    ("clocks.mem", "clocks_memory_mhz"),
    ("pcie.link.gen.current", "pcie_link_gen_current"),
    ("pcie.link.width.current", "pcie_link_width_current"),
    ("clocks_throttle_reasons.active", "throttle_reasons"),
]

BASE_QUERY_FIELDS = [
    ("index", "gpu_index"),
    ("name", "gpu_name"),
    ("driver_version", "driver_version"),
    ("temperature.gpu", "temperature_gpu_c"),
    ("power.draw", "power_draw_w"),
    ("power.limit", "power_limit_w"),
    ("utilization.gpu", "utilization_gpu_percent"),
    ("utilization.memory", "utilization_memory_percent"),
    ("memory.total", "memory_total_mb"),
    ("memory.used", "memory_used_mb"),
    ("memory.free", "memory_free_mb"),
    ("clocks.gr", "clocks_graphics_mhz"),
    ("clocks.mem", "clocks_memory_mhz"),
]


def _optional_psutil() -> Any | None:
    try:
        import psutil  # type: ignore
    except ImportError:
        return None
    return psutil


def _parse_cuda_version(nvidia_smi_output: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_smi_output)
    if match:
        return match.group(1)
    return None


class TelemetryCollector:
    """Collect telemetry in a background thread while a benchmark runs."""

    def __init__(
        self,
        output_path: Path,
        interval_seconds: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.logger = logger or logging.getLogger(__name__)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._file: Any | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._nvidia_smi_path = shutil.which("nvidia-smi")
        self._cuda_version_from_smi = self._load_cuda_version()
        self._psutil = _optional_psutil()
        self.samples_written = 0
        self.gpu_samples_written = 0
        self.warning: str | None = None

    @property
    def nvidia_smi_available(self) -> bool:
        """Return whether nvidia-smi was found on PATH."""
        return self._nvidia_smi_path is not None

    def start(self) -> None:
        """Start background sampling and create the CSV file immediately."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TELEMETRY_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        if not self.nvidia_smi_available:
            self.warning = "nvidia-smi not found; GPU telemetry unavailable."
            self.logger.warning(self.warning)
        self._thread = threading.Thread(target=self._run, name="telemetry-collector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and close the CSV file."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=max(self.interval_seconds + 5, 6))
        if self._file:
            self._file.flush()
            self._file.close()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.sample_once()
            except Exception as exc:  # pragma: no cover - defensive thread boundary
                self.logger.warning("Telemetry sample failed: %s", exc)
            if self._stop_event.wait(self.interval_seconds):
                break

    def sample_once(self) -> None:
        """Collect one telemetry sample and append it to the CSV."""
        if not self._writer:
            raise RuntimeError("Telemetry collector has not been started")

        timestamp = utc_now_iso()
        system_values = self._system_sample()
        gpu_rows = self._gpu_sample()

        if not gpu_rows:
            row = self._empty_row(timestamp)
            row.update(system_values)
            self._writer.writerow(row)
            self.samples_written += 1
        else:
            for gpu_row in gpu_rows:
                row = self._empty_row(timestamp)
                row.update(gpu_row)
                row.update(system_values)
                self._writer.writerow(row)
                self.samples_written += 1
                self.gpu_samples_written += 1
        if self._file:
            self._file.flush()

    def _empty_row(self, timestamp: str) -> dict[str, Any]:
        return {field: "" for field in TELEMETRY_FIELDS} | {"timestamp": timestamp}

    def _system_sample(self) -> dict[str, Any]:
        if not self._psutil:
            return {}
        ram = self._psutil.virtual_memory()
        return {
            "cpu_percent": self._psutil.cpu_percent(interval=None),
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "ram_used_gb": round(ram.used / (1024**3), 2),
            "ram_percent": ram.percent,
        }

    def _load_cuda_version(self) -> str | None:
        if not self._nvidia_smi_path:
            return None
        code, stdout, _ = run_command([self._nvidia_smi_path], 5)
        if code != 0:
            return None
        return _parse_cuda_version(stdout)

    def _gpu_sample(self) -> list[dict[str, Any]]:
        if not self._nvidia_smi_path:
            return []
        for query_fields in (FULL_QUERY_FIELDS, BASE_QUERY_FIELDS):
            rows = self._query_gpu_fields(query_fields)
            if rows is not None:
                return rows
        self.warning = "nvidia-smi query failed; GPU telemetry unavailable for this sample."
        self.logger.warning(self.warning)
        return []

    def _query_gpu_fields(self, fields: list[tuple[str, str]]) -> list[dict[str, Any]] | None:
        query = ",".join(field for field, _ in fields)
        code, stdout, stderr = run_command(
            [
                self._nvidia_smi_path or "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            5,
        )
        if code != 0:
            self.logger.debug("nvidia-smi query failed: %s", stderr)
            return None
        if not stdout:
            return []

        output_names = [output_name for _, output_name in fields]
        rows: list[dict[str, Any]] = []
        for parsed_row in csv.reader(stdout.splitlines()):
            row = {name: value.strip() for name, value in zip(output_names, parsed_row, strict=False)}
            row["cuda_version"] = self._cuda_version_from_smi or ""
            rows.append(row)
        return rows
