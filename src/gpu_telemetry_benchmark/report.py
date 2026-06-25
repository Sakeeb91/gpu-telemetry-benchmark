"""Markdown and machine-readable validation report generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .utils import max_numeric, mean_numeric, read_json, safe_float, write_json


def _read_telemetry(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _fmt(value: Any, suffix: str = "", precision: int = 2) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.{precision}f}{suffix}"


def _numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = safe_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def summarize_telemetry(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Calculate report-friendly telemetry summary values."""
    gpu_rows = [row for row in rows if row.get("gpu_index", "").strip()]
    gpu_temps = _numeric_values(gpu_rows, "temperature_gpu_c")
    return {
        "sample_count": len(rows),
        "gpu_sample_count": len(gpu_rows),
        "gpu_telemetry_available": bool(gpu_rows),
        "first_sample_timestamp": rows[0].get("timestamp") if rows else None,
        "last_sample_timestamp": rows[-1].get("timestamp") if rows else None,
        "peak_gpu_temperature_c": max_numeric(row.get("temperature_gpu_c") for row in gpu_rows),
        "gpu_temperature_start_c": gpu_temps[0] if gpu_temps else None,
        "gpu_temperature_end_c": gpu_temps[-1] if gpu_temps else None,
        "gpu_temperature_delta_c": (gpu_temps[-1] - gpu_temps[0]) if len(gpu_temps) >= 2 else None,
        "average_gpu_utilization_percent": mean_numeric(
            row.get("utilization_gpu_percent") for row in gpu_rows
        ),
        "peak_gpu_memory_used_mb": max_numeric(row.get("memory_used_mb") for row in gpu_rows),
        "average_power_draw_w": mean_numeric(row.get("power_draw_w") for row in gpu_rows),
        "average_cpu_percent": mean_numeric(row.get("cpu_percent") for row in rows),
        "peak_ram_percent": max_numeric(row.get("ram_percent") for row in rows),
    }


def evaluate_run(
    run_dir: Path,
    system_info: dict[str, Any],
    results: dict[str, Any],
    telemetry_rows: list[dict[str, str]],
    *,
    temperature_threshold_c: float = 85.0,
    min_gpu_utilization_percent: float = 50.0,
) -> tuple[str, list[dict[str, str]], list[str]]:
    """Evaluate pass/fail criteria and return classification, criteria, anomalies."""
    telemetry_summary = summarize_telemetry(telemetry_rows)
    criteria: list[dict[str, str]] = []
    anomalies: list[str] = []

    def add(name: str, status: str, evidence: str) -> None:
        criteria.append({"criterion": name, "status": status, "evidence": evidence})
        if status in {"WARN", "FAIL"}:
            anomalies.append(f"{name}: {evidence}")

    requested_device = results.get("device_requested")
    used_device = results.get("device_used")
    status = results.get("status")

    if requested_device == "cuda" and used_device != "cuda":
        device_status = "FAIL"
        device_evidence = f"requested={requested_device}, used={used_device}"
    elif requested_device == "auto" and used_device == "cpu":
        device_status = "WARN"
        device_evidence = "GPU unavailable; CPU fallback used. This is not GPU validation evidence."
    else:
        device_status = "PASS"
        device_evidence = f"requested={requested_device}, used={used_device}"
    add("Device selection matched validation intent", device_status, device_evidence)

    add(
        "Benchmark completed without runtime crash",
        "PASS" if status == "completed" else "FAIL",
        f"status={status}",
    )

    error_type = str(results.get("error_type") or "")
    add(
        "No GPU out-of-memory error",
        "FAIL" if "out_of_memory" in error_type else "PASS",
        error_type or "No OOM error recorded",
    )

    add(
        "Telemetry samples were captured",
        "PASS" if telemetry_summary["sample_count"] > 0 else "FAIL",
        f"{telemetry_summary['sample_count']} sample(s)",
    )

    if used_device == "cuda":
        gpu_telemetry_status = "PASS" if telemetry_summary["gpu_telemetry_available"] else "WARN"
        gpu_telemetry_evidence = (
            f"{telemetry_summary['gpu_sample_count']} GPU sample(s)"
            if telemetry_summary["gpu_telemetry_available"]
            else "CUDA workload ran without nvidia-smi GPU telemetry"
        )
    else:
        gpu_telemetry_status = "PASS"
        gpu_telemetry_evidence = "Not required for CPU run"
    add("GPU telemetry was captured for CUDA runs", gpu_telemetry_status, gpu_telemetry_evidence)

    peak_temp = telemetry_summary["peak_gpu_temperature_c"]
    if peak_temp is None:
        temp_status = "WARN" if used_device == "cuda" else "PASS"
        temp_evidence = "GPU temperature unavailable" if temp_status == "WARN" else "CPU run or no GPU telemetry"
    elif peak_temp <= temperature_threshold_c:
        temp_status = "PASS"
        temp_evidence = f"Peak {peak_temp:.2f} C <= {temperature_threshold_c:.2f} C"
    else:
        temp_status = "FAIL"
        temp_evidence = f"Peak {peak_temp:.2f} C > {temperature_threshold_c:.2f} C"
    add("GPU temperature stayed below threshold", temp_status, temp_evidence)

    temp_delta = telemetry_summary["gpu_temperature_delta_c"]
    if temp_delta is None:
        runaway_status = "WARN" if used_device == "cuda" else "PASS"
        runaway_evidence = "Insufficient GPU temperature samples" if used_device == "cuda" else "Not applicable for CPU run"
    elif temp_delta > 15 and peak_temp is not None and peak_temp >= temperature_threshold_c - 5:
        runaway_status = "FAIL"
        runaway_evidence = f"Temperature rose {temp_delta:.2f} C and peak approached threshold"
    else:
        runaway_status = "PASS"
        runaway_evidence = f"Temperature delta {temp_delta:.2f} C"
    add("No thermal runaway trend detected", runaway_status, runaway_evidence)

    avg_util = telemetry_summary["average_gpu_utilization_percent"]
    if used_device == "cuda":
        if avg_util is None:
            util_status = "WARN"
            util_evidence = "GPU utilization telemetry unavailable"
        elif avg_util >= min_gpu_utilization_percent:
            util_status = "PASS"
            util_evidence = f"Average {avg_util:.2f}% >= {min_gpu_utilization_percent:.2f}%"
        else:
            util_status = "WARN"
            util_evidence = f"Average {avg_util:.2f}% < {min_gpu_utilization_percent:.2f}%"
    else:
        util_status = "PASS"
        util_evidence = "Not required for CPU run"
    add("Average GPU utilization exceeded threshold for GPU runs", util_status, util_evidence)

    telemetry_path = run_dir / "telemetry.csv"
    add(
        "Telemetry file was created",
        "PASS" if telemetry_path.exists() else "FAIL",
        str(telemetry_path),
    )

    system_info_path = run_dir / "system_info.json"
    add(
        "System info file was created",
        "PASS" if system_info_path.exists() and bool(system_info) else "FAIL",
        str(system_info_path),
    )

    if any(item["status"] == "FAIL" for item in criteria):
        classification = "FAIL"
    elif any(item["status"] == "WARN" for item in criteria):
        classification = "PASS WITH WARNINGS"
    else:
        classification = "PASS"
    return classification, criteria, anomalies


def generate_report(
    run_dir: Path,
    *,
    temperature_threshold_c: float = 85.0,
    min_gpu_utilization_percent: float = 50.0,
) -> Path:
    """Generate ``validation_report.md`` and ``validation_summary.json`` from run artifacts."""
    run_dir = run_dir.resolve()
    system_info_path = run_dir / "system_info.json"
    results_path = run_dir / "benchmark_results.json"
    telemetry_path = run_dir / "telemetry.csv"

    system_info = read_json(system_info_path) if system_info_path.exists() else {}
    results = read_json(results_path) if results_path.exists() else {}
    telemetry_rows = _read_telemetry(telemetry_path)
    telemetry_summary = summarize_telemetry(telemetry_rows)
    classification, criteria, anomalies = evaluate_run(
        run_dir,
        system_info,
        results,
        telemetry_rows,
        temperature_threshold_c=temperature_threshold_c,
        min_gpu_utilization_percent=min_gpu_utilization_percent,
    )

    workload = results.get("workload", "unknown")
    device_used = results.get("device_used", "unknown")
    throughput = results.get("throughput", {})
    latency = results.get("latency_statistics_ms", {})

    validation_summary = {
        "classification": classification,
        "criteria": criteria,
        "anomalies": anomalies,
        "telemetry_summary": telemetry_summary,
        "thresholds": {
            "temperature_threshold_c": temperature_threshold_c,
            "min_gpu_utilization_percent": min_gpu_utilization_percent,
        },
        "artifact_paths": {
            "system_info": str(system_info_path),
            "benchmark_results": str(results_path),
            "telemetry": str(telemetry_path),
            "validation_report": str(run_dir / "validation_report.md"),
        },
    }
    write_json(run_dir / "validation_summary.json", validation_summary)

    lines = [
        f"# Validation Report: {workload}",
        "",
        "## Executive Summary",
        "",
        f"- Classification: **{classification}**",
        f"- Workload: `{workload}`",
        f"- Device used: `{device_used}`",
        f"- Status: `{results.get('status', 'unknown')}`",
        f"- Duration: {_fmt(results.get('measured_duration_seconds'), ' s')}",
        f"- Primary throughput: {_fmt(throughput.get('value'))} {throughput.get('unit', '')}",
        f"- Engineering assessment: {_assessment_sentence(classification, device_used, telemetry_summary)}",
        "",
        "## Validation Scope",
        "",
        f"- Validation focus: {results.get('validation_focus', 'N/A')}",
        f"- Resource profile: {results.get('resource_profile', 'N/A')}",
        f"- Evidence scope: {_evidence_scope(device_used, telemetry_summary['gpu_telemetry_available'])}",
        f"- Random seed: {results.get('parameters', {}).get('seed', 'N/A')}",
        f"- GPU index selected: {results.get('gpu_index', 'N/A')}",
        "",
        "## System Under Test",
        "",
        f"- Hostname: `{system_info.get('hostname', 'N/A')}`",
        f"- OS: {system_info.get('platform', 'N/A')}",
        f"- Kernel: {system_info.get('kernel_version', 'N/A')}",
        f"- CPU: {system_info.get('cpu_model', 'N/A')}",
        f"- CPU cores: logical={system_info.get('cpu_core_count_logical', 'N/A')}, "
        f"physical={system_info.get('cpu_core_count_physical', 'N/A')}",
        f"- RAM: {system_info.get('ram_total_gb', 'N/A')} GB",
        f"- GPU count from PyTorch: {system_info.get('gpu_count', 'N/A')}",
        f"- GPU names from PyTorch: {system_info.get('gpu_names_detected_by_pytorch', [])}",
        f"- NVIDIA GPU inventory from nvidia-smi: {system_info.get('nvidia_gpu_inventory', [])}",
        f"- NVIDIA driver: {system_info.get('nvidia_driver_version', 'N/A')}",
        "",
        "## Software Stack",
        "",
        f"- Python: {system_info.get('python_version', 'N/A')}",
        f"- PyTorch installed: {system_info.get('pytorch_installed', 'N/A')}",
        f"- PyTorch version: {system_info.get('pytorch_version', 'N/A')}",
        f"- CUDA available from PyTorch: {system_info.get('cuda_available_from_pytorch', 'N/A')}",
        f"- PyTorch CUDA version: {system_info.get('pytorch_cuda_version', 'N/A')}",
        f"- cuDNN version: {system_info.get('cudnn_version', 'N/A')}",
        f"- CUDA version from nvidia-smi: {system_info.get('nvidia_cuda_version_from_smi', 'N/A')}",
        "",
        "## Benchmark Workload",
        "",
        f"- Workload: `{workload}`",
        f"- Device requested: `{results.get('device_requested', 'N/A')}`",
        f"- Device used: `{device_used}`",
        f"- Device label: `{results.get('device_label', 'N/A')}`",
        f"- Parameters: `{results.get('parameters', {})}`",
        f"- Warmup duration: {_fmt(results.get('warmup_seconds'), ' s')}",
        f"- Start time: {results.get('start_time', 'N/A')}",
        f"- End time: {results.get('end_time', 'N/A')}",
        "",
        "## Results Summary",
        "",
        f"- Total iterations: {results.get('total_iterations', 'N/A')}",
        f"- Units processed: {results.get('total_units_processed', 'N/A')} "
        f"{results.get('unit_name', '')}",
        f"- Throughput: {_fmt(throughput.get('value'))} {throughput.get('unit', '')}",
        f"- Average latency: {_fmt(latency.get('average_ms'), ' ms')}",
        f"- p50 latency: {_fmt(latency.get('p50_ms'), ' ms')}",
        f"- p95 latency: {_fmt(latency.get('p95_ms'), ' ms')}",
        f"- p99 latency: {_fmt(latency.get('p99_ms'), ' ms')}",
        f"- Memory metrics: `{results.get('memory_metrics', {})}`",
        f"- Error: {results.get('error_message') or 'None'}",
        "",
        "## Telemetry Summary",
        "",
        f"- Telemetry samples: {telemetry_summary['sample_count']}",
        f"- GPU telemetry samples: {telemetry_summary['gpu_sample_count']}",
        f"- First sample: {telemetry_summary['first_sample_timestamp'] or 'N/A'}",
        f"- Last sample: {telemetry_summary['last_sample_timestamp'] or 'N/A'}",
        f"- Peak GPU temperature: {_fmt(telemetry_summary['peak_gpu_temperature_c'], ' C')}",
        f"- GPU temperature delta: {_fmt(telemetry_summary['gpu_temperature_delta_c'], ' C')}",
        f"- Average GPU utilization: {_fmt(telemetry_summary['average_gpu_utilization_percent'], '%')}",
        f"- Peak GPU memory usage: {_fmt(telemetry_summary['peak_gpu_memory_used_mb'], ' MB')}",
        f"- Average power draw: {_fmt(telemetry_summary['average_power_draw_w'], ' W')}",
        f"- Average CPU utilization: {_fmt(telemetry_summary['average_cpu_percent'], '%')}",
        f"- Peak RAM utilization: {_fmt(telemetry_summary['peak_ram_percent'], '%')}",
        "",
        "## Thresholds",
        "",
        f"- GPU temperature threshold: {_fmt(temperature_threshold_c, ' C')}",
        f"- Minimum average GPU utilization for CUDA runs: {_fmt(min_gpu_utilization_percent, '%')}",
        "- GPU utilization below threshold is a warning because small or memory-bound workloads may not saturate a device.",
        "",
        "## Detected Anomalies",
        "",
    ]

    if anomalies:
        lines.extend(f"- {item}" for item in anomalies)
    else:
        lines.append("- None detected.")

    lines.extend(
        [
            "",
            "## Pass/Fail Criteria",
            "",
            "| Criterion | Status | Evidence |",
            "| --- | --- | --- |",
        ]
    )
    lines.extend(
        f"| {item['criterion']} | {item['status']} | {item['evidence']} |" for item in criteria
    )

    notes = results.get("notes", [])
    recommendations = _recommendations(classification, anomalies, device_used)
    limitations = _limitations(device_used, telemetry_summary["gpu_telemetry_available"])

    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in notes) if notes else lines.append("- None.")

    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in recommendations)

    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in limitations)

    lines.extend(["", "## Next Steps", ""])
    if device_used == "cpu":
        lines.append("- Rerun on an NVIDIA GPU host before presenting this as GPU validation evidence.")
    lines.extend(
        [
            "- Repeat the run with the same parameters to compare variance.",
            "- Add this run to a batch sweep if throughput scaling is the validation target.",
            "- Attach this report and raw telemetry when documenting any observed issue.",
        ]
    )

    report_path = run_dir / "validation_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _recommendations(classification: str, anomalies: list[str], device_used: str) -> list[str]:
    recommendations: list[str] = []
    if classification == "FAIL":
        recommendations.append("Review the recorded error and rerun with smaller workload parameters.")
    if device_used == "cpu":
        recommendations.append("Run on an NVIDIA GPU host to validate GPU telemetry, power, thermal, and utilization behavior.")
    if anomalies:
        recommendations.append("Investigate warnings or failures before using the run as a clean validation baseline.")
    if not recommendations:
        recommendations.append("Use this run as a baseline and compare future driver, hardware, or parameter changes against it.")
    return recommendations


def _assessment_sentence(
    classification: str,
    device_used: str,
    telemetry_summary: dict[str, Any],
) -> str:
    if classification == "FAIL":
        return "Run failed one or more validation criteria and should not be used as a clean baseline."
    if device_used == "cpu":
        return "Run is useful for workflow verification only; it is not GPU hardware validation evidence."
    if not telemetry_summary["gpu_telemetry_available"]:
        return "Run completed on CUDA but lacks GPU telemetry, so hardware conclusions are limited."
    if classification == "PASS WITH WARNINGS":
        return "Run completed but warnings require review before using it as a baseline."
    return "Run completed with required artifacts and no detected threshold violations."


def _evidence_scope(device_used: str, gpu_telemetry_available: bool) -> str:
    if device_used == "cpu":
        return "CPU-only software workflow validation; no GPU hardware behavior validated."
    if gpu_telemetry_available:
        return "CUDA workload with synchronized nvidia-smi and system telemetry."
    return "CUDA workload evidence with missing or limited nvidia-smi telemetry."


def _limitations(device_used: str, gpu_telemetry_available: bool) -> list[str]:
    limitations = [
        "Synthetic workloads are useful for controlled validation, but they do not replace application-specific qualification.",
        "Results are specific to the recorded hardware, drivers, software stack, and workload parameters.",
    ]
    if device_used == "cpu":
        limitations.append("CPU mode cannot validate GPU power, thermal, clock, memory, or utilization behavior.")
    if not gpu_telemetry_available:
        limitations.append("GPU telemetry was unavailable or telemetry-limited for this run.")
    return limitations


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for regenerating a report from an existing run directory."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate a validation report from benchmark artifacts.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--temperature-threshold-c", type=float, default=85.0)
    parser.add_argument("--min-gpu-utilization-percent", type=float, default=50.0)
    args = parser.parse_args(argv)
    report_path = generate_report(
        args.run_dir,
        temperature_threshold_c=args.temperature_threshold_c,
        min_gpu_utilization_percent=args.min_gpu_utilization_percent,
    )
    print(report_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
