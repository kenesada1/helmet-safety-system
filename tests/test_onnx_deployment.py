from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> object:
    path = ROOT / "scripts" / "deploy" / f"{name}.py"
    if not path.is_file():
        pytest.fail(f"scripts/{name}.py must exist")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.fail(f"could not load scripts/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_validation_refuses_existing_output_without_force(tmp_path: Path) -> None:
    exporter = _load_script("export_onnx")
    weights = tmp_path / "best.pt"
    output = tmp_path / "model.onnx"
    weights.write_bytes(b"pt")
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--force"):
        exporter.validate_export_request(
            weights=weights,
            output=output,
            imgsz=960,
            opset=17,
            force=False,
        )


def test_export_script_makes_local_src_package_importable() -> None:
    script = ROOT / "scripts" / "deploy" / "export_onnx.py"
    command = (
        "import runpy; "
        f"runpy.run_path({str(script)!r}); "
        "import helmet_safety.inference.opencv"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("imgsz", "opset", "message"),
    [(0, 17, "imgsz"), (960, 0, "opset")],
)
def test_export_validation_rejects_non_positive_dimensions(
    tmp_path: Path, imgsz: int, opset: int, message: str
) -> None:
    exporter = _load_script("export_onnx")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")

    with pytest.raises(ValueError, match=message):
        exporter.validate_export_request(
            weights=weights,
            output=tmp_path / "model.onnx",
            imgsz=imgsz,
            opset=opset,
            force=False,
        )


def test_export_replace_overwrites_existing_file(tmp_path: Path) -> None:
    exporter = _load_script("export_onnx")
    exported = tmp_path / "temporary.onnx"
    output = tmp_path / "model.onnx"
    exported.write_bytes(b"new-model")
    output.write_bytes(b"old-model")

    exporter.replace_exported_file(exported, output)

    assert output.read_bytes() == b"new-model"
    assert not exported.exists()


def test_latency_summary_uses_linear_percentiles_and_measured_fps() -> None:
    benchmark = _load_script("benchmark_backends")

    summary = benchmark.summarize_latencies_ms([10.0, 20.0, 30.0, 40.0])

    assert summary == {
        "iterations": 4,
        "mean_ms": 25.0,
        "p50_ms": 25.0,
        "p95_ms": 38.5,
        "min_ms": 10.0,
        "max_ms": 40.0,
        "fps": 40.0,
    }


def test_markdown_report_renders_results_and_unavailable_backends() -> None:
    benchmark = _load_script("benchmark_backends")
    report = {
        "config": {"batch": 1, "imgsz": 960, "warmup": 20, "iterations": 100},
        "environment": {"gpu": "NVIDIA test GPU", "onnxruntime": "1.2.3"},
        "backends": [
            {
                "backend": "PyTorch CUDA",
                "status": "ok",
                "device": "cuda:0",
                "mean_ms": 10.0,
                "p50_ms": 9.0,
                "p95_ms": 12.0,
                "fps": 100.0,
            },
            {
                "backend": "ONNX Runtime CUDA",
                "status": "unavailable",
                "device": "unavailable",
                "reason": "CUDAExecutionProvider is not installed",
            },
        ],
    }

    markdown = benchmark.render_markdown_report(report)

    assert "| 后端 | 设备 | 平均延迟 | P50 | P95 | FPS |" in markdown
    assert "| PyTorch CUDA | cuda:0 | 10.00 ms | 9.00 ms | 12.00 ms | 100.00 |" in markdown
    assert "| ONNX Runtime CUDA | unavailable | unavailable | unavailable | unavailable | unavailable |" in markdown
    assert "CUDAExecutionProvider is not installed" in markdown
    assert "batch=1、imgsz=960" in markdown


def test_explicit_busy_gpu_override_allows_benchmark() -> None:
    benchmark = _load_script("benchmark_backends")

    reason = benchmark.gpu_skip_reason(
        gpu_available=True,
        gpu_processes=[{"pid": "11364", "process_name": "python.exe"}],
        gpu_check_error=None,
        allow_busy_gpu=True,
    )

    assert reason is None
