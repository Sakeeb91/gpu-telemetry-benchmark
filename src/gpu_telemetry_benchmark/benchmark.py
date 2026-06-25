"""Command-line benchmark runner."""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path
from typing import Any

from .report import generate_report
from .system_info import collect_system_info
from .telemetry import TelemetryCollector
from .utils import (
    detect_oom_error,
    ensure_output_dir,
    latency_stats_ms,
    parse_int_list,
    setup_logging,
    utc_now_iso,
    validate_non_negative,
    validate_positive,
    write_json,
)
from .workloads import (
    create_workload,
    cuda_memory_metrics,
    empty_cuda_cache_if_needed,
    reset_peak_memory,
    resolve_device,
    synchronize_if_needed,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run PyTorch AI/HPC-style benchmarks with telemetry collection."
    )
    parser.add_argument("--workload", required=True, help="matmul, conv2d, or transformer")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    parser.add_argument("--telemetry-interval", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--sweep-batch-sizes", default=None)
    parser.add_argument("--temperature-threshold-c", type=float, default=85.0)
    parser.add_argument("--min-gpu-utilization-percent", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate common CLI arguments."""
    validate_positive("--duration-seconds", args.duration_seconds)
    validate_non_negative("--warmup-seconds", args.warmup_seconds)
    validate_positive("--telemetry-interval", args.telemetry_interval)
    validate_positive("--matrix-size", args.matrix_size)
    validate_positive("--batch-size", args.batch_size)
    validate_positive("--input-size", args.input_size)
    validate_positive("--sequence-length", args.sequence_length)
    validate_positive("--hidden-dim", args.hidden_dim)
    validate_positive("--num-heads", args.num_heads)
    validate_positive("--num-layers", args.num_layers)
    validate_positive("--temperature-threshold-c", args.temperature_threshold_c)
    validate_non_negative("--min-gpu-utilization-percent", args.min_gpu_utilization_percent)


def _parameters_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "matrix_size": args.matrix_size,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "sequence_length": args.sequence_length,
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "telemetry_interval": args.telemetry_interval,
        "temperature_threshold_c": args.temperature_threshold_c,
        "min_gpu_utilization_percent": args.min_gpu_utilization_percent,
    }


def _base_results(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    return {
        "workload": args.workload,
        "device_requested": args.device,
        "device_used": None,
        "parameters": _parameters_from_args(args),
        "output_dir": str(run_dir),
        "start_time": None,
        "end_time": None,
        "requested_duration_seconds": args.duration_seconds,
        "measured_duration_seconds": 0.0,
        "warmup_seconds": args.warmup_seconds,
        "total_iterations": 0,
        "total_units_processed": 0,
        "unit_name": None,
        "latency_statistics_ms": latency_stats_ms([]),
        "throughput": {},
        "workload_metrics": {},
        "memory_metrics": {},
        "telemetry": {},
        "status": "failed",
        "error_type": None,
        "error_message": None,
        "notes": [],
    }


def run_single_benchmark(
    args: argparse.Namespace,
    *,
    output_dir: Path | None = None,
    overwrite: bool | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run one benchmark and write all artifacts into an output directory."""
    logger = logger or logging.getLogger(__name__)
    run_dir = ensure_output_dir(output_dir or args.output_dir, args.overwrite if overwrite is None else overwrite)
    results = _base_results(args, run_dir)
    system_info = collect_system_info()
    write_json(run_dir / "system_info.json", system_info)

    telemetry = TelemetryCollector(
        run_dir / "telemetry.csv",
        interval_seconds=args.telemetry_interval,
        logger=logger,
    )
    device = None
    start_counter = time.perf_counter()
    results["start_time"] = utc_now_iso()
    telemetry.start()

    try:
        resolution = resolve_device(args.device)
        device = resolution.device
        results["device_used"] = resolution.device_type
        results["cuda_available"] = resolution.cuda_available
        results["fallback_used"] = resolution.fallback_used
        results["notes"].extend(resolution.notes)

        workload = create_workload(
            args.workload,
            device,
            matrix_size=args.matrix_size,
            batch_size=args.batch_size,
            input_size=args.input_size,
            sequence_length=args.sequence_length,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
        )
        results["parameters"].update(workload.parameters)
        results["unit_name"] = workload.unit_name

        _run_warmup(workload, device, args.warmup_seconds, logger)
        reset_peak_memory(device)

        latencies: list[float] = []
        total_units = 0
        iterations = 0
        benchmark_start = time.perf_counter()
        while time.perf_counter() - benchmark_start < args.duration_seconds:
            synchronize_if_needed(device)
            iteration_start = time.perf_counter()
            units = workload.run_iteration()
            synchronize_if_needed(device)
            iteration_end = time.perf_counter()
            latencies.append(iteration_end - iteration_start)
            iterations += 1
            total_units += units

        measured_duration = time.perf_counter() - benchmark_start
        workload_metrics = workload.extra_metrics(iterations, measured_duration, total_units)
        results.update(
            {
                "measured_duration_seconds": measured_duration,
                "total_iterations": iterations,
                "total_units_processed": total_units,
                "latency_statistics_ms": latency_stats_ms(latencies),
                "throughput": {
                    "metric_name": workload.throughput_name,
                    "value": workload_metrics.get(workload.throughput_name),
                    "unit": workload.throughput_unit,
                },
                "workload_metrics": workload_metrics,
                "memory_metrics": cuda_memory_metrics(device),
                "status": "completed",
            }
        )
    except KeyboardInterrupt:
        results["status"] = "interrupted"
        results["error_type"] = "KeyboardInterrupt"
        results["error_message"] = "Benchmark interrupted by user."
        logger.warning(results["error_message"])
    except Exception as exc:
        if device is not None and detect_oom_error(exc):
            empty_cuda_cache_if_needed(device)
            results["error_type"] = "cuda_out_of_memory"
        else:
            results["error_type"] = type(exc).__name__
        results["error_message"] = str(exc)
        results["status"] = "failed"
        logger.error("Benchmark failed: %s", exc)
    finally:
        telemetry.stop()
        results["end_time"] = utc_now_iso()
        results["wall_time_seconds"] = time.perf_counter() - start_counter
        results["telemetry"] = {
            "path": str(run_dir / "telemetry.csv"),
            "samples_written": telemetry.samples_written,
            "gpu_samples_written": telemetry.gpu_samples_written,
            "nvidia_smi_available": telemetry.nvidia_smi_available,
            "warning": telemetry.warning,
        }
        if telemetry.warning:
            results["notes"].append(telemetry.warning)
        write_json(run_dir / "benchmark_results.json", results)
        report_path = generate_report(
            run_dir,
            temperature_threshold_c=args.temperature_threshold_c,
            min_gpu_utilization_percent=args.min_gpu_utilization_percent,
        )
        logger.info("Wrote validation report: %s", report_path)

    return results


def _run_warmup(workload: Any, device: Any, warmup_seconds: float, logger: logging.Logger) -> None:
    if warmup_seconds <= 0:
        return
    logger.info("Running %.2fs warmup", warmup_seconds)
    end_time = time.perf_counter() + warmup_seconds
    while time.perf_counter() < end_time:
        workload.run_iteration()
    synchronize_if_needed(device)


def run_sweep(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    """Run a batch-size sweep and write an aggregate JSON/Markdown summary."""
    batch_sizes = parse_int_list(args.sweep_batch_sizes)
    root_dir = ensure_output_dir(args.output_dir, args.overwrite)
    sweep_results: dict[str, Any] = {
        "workload": args.workload,
        "device_requested": args.device,
        "batch_sizes": batch_sizes,
        "start_time": utc_now_iso(),
        "end_time": None,
        "runs": [],
        "status": "completed",
    }

    for batch_size in batch_sizes:
        run_args = copy.copy(args)
        run_args.batch_size = batch_size
        run_args.sweep_batch_sizes = None
        run_dir = root_dir / f"batch_size_{batch_size}"
        logger.info("Running sweep batch_size=%s", batch_size)
        result = run_single_benchmark(run_args, output_dir=run_dir, overwrite=False, logger=logger)
        sweep_results["runs"].append(
            {
                "batch_size": batch_size,
                "status": result.get("status"),
                "run_dir": str(run_dir),
                "throughput": result.get("throughput", {}),
                "latency_statistics_ms": result.get("latency_statistics_ms", {}),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
        )
        if result.get("status") != "completed":
            sweep_results["status"] = "completed_with_failures"

    sweep_results["end_time"] = utc_now_iso()
    write_json(root_dir / "sweep_results.json", sweep_results)
    _write_sweep_report(root_dir, sweep_results)
    return sweep_results


def _write_sweep_report(root_dir: Path, sweep_results: dict[str, Any]) -> None:
    lines = [
        f"# Batch Sweep Report: {sweep_results['workload']}",
        "",
        f"- Status: `{sweep_results['status']}`",
        f"- Device requested: `{sweep_results['device_requested']}`",
        f"- Batch sizes: `{sweep_results['batch_sizes']}`",
        "",
        "| Batch size | Status | Throughput | p95 latency ms | Run directory |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in sweep_results["runs"]:
        throughput = run.get("throughput", {})
        latency = run.get("latency_statistics_ms", {})
        throughput_value = throughput.get("value")
        throughput_text = "N/A" if throughput_value is None else f"{throughput_value:.2f} {throughput.get('unit', '')}"
        p95 = latency.get("p95_ms")
        p95_text = "N/A" if p95 is None else f"{p95:.2f}"
        lines.append(
            f"| {run['batch_size']} | {run['status']} | {throughput_text} | "
            f"{p95_text} | `{run['run_dir']}` |"
        )
    lines.extend(
        [
            "",
            "Each subdirectory contains its own system inventory, telemetry CSV, benchmark results JSON, and validation report.",
        ]
    )
    (root_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Benchmark CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = setup_logging(args.verbose)

    try:
        validate_args(args)
        if args.sweep_batch_sizes:
            result = run_sweep(args, logger)
            return 0 if result["status"] == "completed" else 1
        result = run_single_benchmark(args, logger=logger)
        return 0 if result.get("status") == "completed" else 1
    except Exception as exc:
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
