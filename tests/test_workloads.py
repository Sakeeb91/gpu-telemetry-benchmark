import pytest

from gpu_telemetry_benchmark.workloads import create_workload


def test_invalid_workload_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Invalid workload"):
        create_workload("not-a-workload", device="cpu")


def test_matmul_workload_initializes_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    workload = create_workload("matmul", torch.device("cpu"), matrix_size=8)

    units = workload.run_iteration()

    assert workload.name == "matmul"
    assert units == 1


def test_conv2d_workload_initializes_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    workload = create_workload("conv2d", torch.device("cpu"), batch_size=1, input_size=16)

    units = workload.run_iteration()

    assert workload.name == "conv2d"
    assert units == 1


def test_transformer_workload_initializes_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    workload = create_workload(
        "transformer",
        torch.device("cpu"),
        batch_size=1,
        sequence_length=4,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
    )

    units = workload.run_iteration()

    assert workload.name == "transformer"
    assert units == 4
