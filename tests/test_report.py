import csv
import json
from pathlib import Path

from gpu_telemetry_benchmark.report import generate_report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_generation_from_fake_cpu_run(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "system_info.json",
        {
            "hostname": "unit-test-host",
            "platform": "TestOS",
            "kernel_version": "test-kernel",
            "cpu_model": "Test CPU",
            "cpu_core_count_logical": 8,
            "cpu_core_count_physical": 4,
            "ram_total_gb": 16,
            "gpu_count": 0,
            "gpu_names_detected_by_pytorch": [],
            "nvidia_driver_version": None,
            "python_version": "3.11",
            "pytorch_installed": True,
            "pytorch_version": "2.x",
            "cuda_available_from_pytorch": False,
            "pytorch_cuda_version": None,
            "cudnn_version": None,
            "nvidia_cuda_version_from_smi": None,
        },
    )
    _write_json(
        tmp_path / "benchmark_results.json",
        {
            "workload": "matmul",
            "device_requested": "auto",
            "device_used": "cpu",
            "parameters": {"matrix_size": 8},
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-01T00:00:02+00:00",
            "requested_duration_seconds": 1,
            "measured_duration_seconds": 1.0,
            "warmup_seconds": 0,
            "total_iterations": 10,
            "total_units_processed": 10,
            "unit_name": "matrix_multiplications",
            "latency_statistics_ms": {
                "average_ms": 1.0,
                "p50_ms": 1.0,
                "p95_ms": 1.2,
                "p99_ms": 1.3,
            },
            "throughput": {
                "metric_name": "matrix_multiplications_per_second",
                "value": 10.0,
                "unit": "matmuls/sec",
            },
            "memory_metrics": {},
            "status": "completed",
            "error_type": None,
            "error_message": None,
            "notes": ["Synthetic pytest fixture."],
        },
    )
    with (tmp_path / "telemetry.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["timestamp", "cpu_percent", "ram_percent"])
        writer.writeheader()
        writer.writerow(
            {
                "timestamp": "2026-01-01T00:00:01+00:00",
                "cpu_percent": "12.5",
                "ram_percent": "40.0",
            }
        )

    report_path = generate_report(tmp_path)

    report_text = report_path.read_text(encoding="utf-8")
    assert "# Validation Report: matmul" in report_text
    assert "PASS WITH WARNINGS" in report_text
    assert "CPU fallback used" in report_text
