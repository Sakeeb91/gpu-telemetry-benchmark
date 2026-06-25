"""Shared helpers for benchmark execution and reporting."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence


LOGGER_NAME = "gpu_telemetry_benchmark"


def utc_now_iso() -> str:
    """Return a compact UTC timestamp suitable for structured logs."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure package logging and return the project logger."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(LOGGER_NAME)


def run_command(args: Sequence[str], timeout_seconds: float = 10) -> tuple[int, str, str]:
    """Run a command and return ``(return_code, stdout, stderr)`` without raising."""
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or f"Timed out after {timeout_seconds}s"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def ensure_output_dir(path: Path, overwrite: bool = False) -> Path:
    """Create a run output directory, refusing to overwrite existing files by default."""
    path = path.expanduser().resolve()
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {path}")
        has_contents = any(path.iterdir())
        if has_contents:
            if not overwrite:
                raise FileExistsError(
                    f"Output directory already exists and is not empty: {path}. "
                    "Choose a new --output-dir or pass --overwrite."
                )
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON file with deterministic indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_int_list(raw: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid integer in list: {item!r}") from exc
        if value <= 0:
            raise ValueError(f"Batch sizes must be positive integers, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one integer value is required")
    return values


def percentile(sorted_values: Sequence[float], percentile_value: float) -> float:
    """Return a percentile using linear interpolation."""
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile_value
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def latency_stats_ms(latencies_seconds: Sequence[float]) -> dict[str, float | int]:
    """Compute latency statistics in milliseconds."""
    if not latencies_seconds:
        return {
            "count": 0,
            "average_ms": math.nan,
            "p50_ms": math.nan,
            "p95_ms": math.nan,
            "p99_ms": math.nan,
            "min_ms": math.nan,
            "max_ms": math.nan,
        }

    latencies_ms = sorted(value * 1000 for value in latencies_seconds)
    return {
        "count": len(latencies_ms),
        "average_ms": mean(latencies_ms),
        "p50_ms": median(latencies_ms),
        "p95_ms": percentile(latencies_ms, 0.95),
        "p99_ms": percentile(latencies_ms, 0.99),
        "min_ms": latencies_ms[0],
        "max_ms": latencies_ms[-1],
    }


def safe_float(value: Any) -> float | None:
    """Convert telemetry values to float, returning None for blanks and N/A values."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean_numeric(values: Iterable[Any]) -> float | None:
    """Average values that can be safely converted to float."""
    numeric_values = [value for value in (safe_float(item) for item in values) if value is not None]
    if not numeric_values:
        return None
    return mean(numeric_values)


def max_numeric(values: Iterable[Any]) -> float | None:
    """Return the maximum of values that can be safely converted to float."""
    numeric_values = [value for value in (safe_float(item) for item in values) if value is not None]
    if not numeric_values:
        return None
    return max(numeric_values)


def bytes_to_gb(byte_count: int | float | None) -> float | None:
    """Convert bytes to GiB-like decimal GB for readable inventory output."""
    if byte_count is None:
        return None
    return round(float(byte_count) / (1024**3), 2)


def detect_oom_error(exc: BaseException) -> bool:
    """Detect CUDA or CPU out-of-memory errors from exception text."""
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def validate_positive(name: str, value: int | float) -> None:
    """Raise a clear error when a numeric CLI parameter is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(name: str, value: int | float) -> None:
    """Raise a clear error when a numeric CLI parameter is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
