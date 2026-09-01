from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E4_WEIGHTS = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "m45_yolo11s_e75_960_001"
    / "weights"
    / "best.pt"
)


def _load_script(name: str) -> object:
    path = PROJECT_ROOT / "scripts" / "inference" / name
    assert path.is_file(), f"M5 CLI script is missing: {path}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_cli_defaults_to_fixed_e4_inference_protocol(tmp_path: Path) -> None:
    script = _load_script("infer_image.py")
    args = script.build_parser().parse_args(
        ["--source", str(tmp_path / "input.jpg"), "--output", str(tmp_path / "out")]
    )

    assert args.weights.resolve() == E4_WEIGHTS.resolve()
    assert args.device == "0"
    assert args.imgsz == 960
    assert args.conf == 0.25
    assert args.iou == 0.70
    assert args.max_det == 300
    assert not args.force
    assert not args.show


def test_video_cli_exposes_stride_max_frames_and_overwrite_controls() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "inference" / "infer_video.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for option in (
        "--source",
        "--output",
        "--weights",
        "--device",
        "--imgsz",
        "--conf",
        "--iou",
        "--max-det",
        "--frame-stride",
        "--max-frames",
        "--show",
        "--force",
    ):
        assert option in result.stdout

