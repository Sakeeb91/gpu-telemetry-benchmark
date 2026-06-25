from gpu_telemetry_benchmark.system_info import collect_system_info


def test_collect_system_info_returns_dictionary() -> None:
    info = collect_system_info()

    assert isinstance(info, dict)
    assert info["timestamp"]
    assert info["python_version"]
    assert "cuda_available_from_pytorch" in info
    assert "gpu_count" in info
