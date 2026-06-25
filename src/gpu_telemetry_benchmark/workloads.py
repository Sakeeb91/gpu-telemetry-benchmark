"""Self-contained PyTorch benchmark workloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_torch() -> Any:
    """Import PyTorch and raise a clear runtime error when it is missing."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is not installed. Install project dependencies with "
            "`python -m pip install -e .` or install a platform-specific torch wheel."
        ) from exc
    return torch


@dataclass(frozen=True)
class DeviceResolution:
    """Resolved benchmark device and notes for reporting."""

    device: Any
    device_type: str
    requested: str
    cuda_available: bool
    fallback_used: bool
    notes: list[str]


def resolve_device(requested: str) -> DeviceResolution:
    """Resolve auto/cpu/cuda device selection."""
    torch = require_torch()
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device {requested!r}; expected auto, cpu, or cuda")

    cuda_available = bool(torch.cuda.is_available())
    notes: list[str] = []
    if requested == "cpu":
        return DeviceResolution(torch.device("cpu"), "cpu", requested, cuda_available, False, notes)
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested with --device cuda, but torch.cuda.is_available() is false.")
        return DeviceResolution(torch.device("cuda"), "cuda", requested, cuda_available, False, notes)
    if cuda_available:
        return DeviceResolution(torch.device("cuda"), "cuda", requested, cuda_available, False, notes)

    notes.append("GPU unavailable; CPU fallback used.")
    return DeviceResolution(torch.device("cpu"), "cpu", requested, cuda_available, True, notes)


def synchronize_if_needed(device: Any) -> None:
    """Synchronize CUDA work before timing boundaries when needed."""
    if getattr(device, "type", str(device)) == "cuda":
        torch = require_torch()
        torch.cuda.synchronize(device)


def reset_peak_memory(device: Any) -> None:
    """Reset CUDA peak memory counters when available."""
    if getattr(device, "type", str(device)) == "cuda":
        torch = require_torch()
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_metrics(device: Any) -> dict[str, float]:
    """Return CUDA memory metrics in MiB when running on CUDA."""
    if getattr(device, "type", str(device)) != "cuda":
        return {}
    torch = require_torch()
    return {
        "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024**2), 2),
        "cuda_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / (1024**2), 2),
        "cuda_max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
        "cuda_max_memory_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024**2), 2),
    }


def empty_cuda_cache_if_needed(device: Any) -> None:
    """Release cached CUDA memory after failed runs when possible."""
    if getattr(device, "type", str(device)) == "cuda":
        torch = require_torch()
        torch.cuda.empty_cache()


class BenchmarkWorkload:
    """Base class for benchmark workloads."""

    name = "base"
    unit_name = "units"
    throughput_name = "units_per_second"
    throughput_unit = "units/sec"
    flops_per_iteration: int | None = None

    def __init__(self, device: Any) -> None:
        self.device = device

    @property
    def parameters(self) -> dict[str, Any]:
        """Return workload parameters for results metadata."""
        return {}

    def run_iteration(self) -> int:
        """Run one workload iteration and return units processed."""
        raise NotImplementedError

    def extra_metrics(self, iterations: int, elapsed_seconds: float, total_units: int) -> dict[str, Any]:
        """Return workload-specific aggregate metrics."""
        metrics: dict[str, Any] = {
            "iterations_per_second": iterations / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            self.throughput_name: total_units / elapsed_seconds if elapsed_seconds > 0 else 0.0,
            "throughput_unit": self.throughput_unit,
        }
        if self.flops_per_iteration and elapsed_seconds > 0:
            flops_per_second = self.flops_per_iteration * iterations / elapsed_seconds
            metrics["approx_flops_per_second"] = flops_per_second
            metrics["approx_tflops"] = flops_per_second / 1e12
        return metrics


class MatmulWorkload(BenchmarkWorkload):
    """Repeated dense matrix multiplication."""

    name = "matmul"
    unit_name = "matrix_multiplications"
    throughput_name = "matrix_multiplications_per_second"
    throughput_unit = "matmuls/sec"

    def __init__(self, device: Any, matrix_size: int = 4096, dtype: str = "float32") -> None:
        super().__init__(device)
        if matrix_size <= 0:
            raise ValueError("matrix_size must be positive")
        torch = require_torch()
        self.matrix_size = matrix_size
        self.dtype_name = dtype
        self.dtype = getattr(torch, dtype)
        self.flops_per_iteration = 2 * matrix_size**3
        self.a = torch.randn(matrix_size, matrix_size, device=device, dtype=self.dtype)
        self.b = torch.randn(matrix_size, matrix_size, device=device, dtype=self.dtype)
        self._last_output: Any | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        return {"matrix_size": self.matrix_size, "dtype": self.dtype_name}

    def run_iteration(self) -> int:
        torch = require_torch()
        with torch.inference_mode():
            self._last_output = self.a @ self.b
        return 1


class Conv2DWorkload(BenchmarkWorkload):
    """Small CNN inference-style workload."""

    name = "conv2d"
    unit_name = "samples"
    throughput_name = "samples_per_second"
    throughput_unit = "samples/sec"

    def __init__(self, device: Any, batch_size: int = 32, input_size: int = 224) -> None:
        super().__init__(device)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        torch = require_torch()
        nn = torch.nn
        self.batch_size = batch_size
        self.input_size = input_size
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1000),
        ).to(device)
        self.model.eval()
        self.input_tensor = torch.randn(batch_size, 3, input_size, input_size, device=device)
        self._last_output: Any | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        return {"batch_size": self.batch_size, "input_size": self.input_size}

    def run_iteration(self) -> int:
        torch = require_torch()
        with torch.inference_mode():
            self._last_output = self.model(self.input_tensor)
        return self.batch_size


class TransformerWorkload(BenchmarkWorkload):
    """Lightweight transformer encoder workload without external model downloads."""

    name = "transformer"
    unit_name = "tokens"
    throughput_name = "tokens_per_second"
    throughput_unit = "tokens/sec"

    def __init__(
        self,
        device: Any,
        batch_size: int = 8,
        sequence_length: int = 512,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__(device)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        torch = require_torch()
        nn = torch.nn
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.model = nn.TransformerEncoder(layer, num_layers=num_layers).to(device)
        self.model.eval()
        self.input_tensor = torch.randn(batch_size, sequence_length, hidden_dim, device=device)
        self._last_output: Any | None = None

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }

    def run_iteration(self) -> int:
        torch = require_torch()
        with torch.inference_mode():
            self._last_output = self.model(self.input_tensor)
        return self.batch_size * self.sequence_length

    def extra_metrics(self, iterations: int, elapsed_seconds: float, total_units: int) -> dict[str, Any]:
        metrics = super().extra_metrics(iterations, elapsed_seconds, total_units)
        if elapsed_seconds > 0:
            metrics["sequences_per_second"] = (iterations * self.batch_size) / elapsed_seconds
        return metrics


def create_workload(
    name: str,
    device: Any,
    *,
    matrix_size: int = 4096,
    batch_size: int = 32,
    input_size: int = 224,
    sequence_length: int = 512,
    hidden_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 2,
) -> BenchmarkWorkload:
    """Construct a named benchmark workload."""
    normalized = name.lower()
    if normalized == "matmul":
        return MatmulWorkload(device=device, matrix_size=matrix_size)
    if normalized == "conv2d":
        return Conv2DWorkload(device=device, batch_size=batch_size, input_size=input_size)
    if normalized == "transformer":
        return TransformerWorkload(
            device=device,
            batch_size=batch_size,
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    raise ValueError(f"Invalid workload {name!r}; expected one of: matmul, conv2d, transformer")
