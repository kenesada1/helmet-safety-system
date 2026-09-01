from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
E4_WEIGHTS = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"


def _script():
    path = PROJECT_ROOT / "scripts" / "inference" / "track_video.py"
    assert path.is_file(), f"M6 CLI script is missing: {path}"
    spec = importlib.util.spec_from_file_location("track_video", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tracking_cli_defaults_match_fixed_m6_baseline(tmp_path: Path) -> None:
    args = _script().build_parser().parse_args(
        ["--source", str(tmp_path / "input.mp4"), "--output", str(tmp_path / "output.mp4")]
    )

    assert args.weights.resolve() == E4_WEIGHTS.resolve()
    assert args.device == "0"
    assert args.imgsz == 960
    assert args.conf == 0.10
    assert args.iou == 0.70
    assert args.max_det == 300
    assert args.tracker == "bytetrack"
    assert args.tracker_config is None
    assert args.frame_stride == 1
    assert args.max_frames is None
    assert args.lost_ttl_seconds == 1.0
    assert args.fallback_fps == 30.0
    assert not args.jsonl
    assert not args.show
    assert not args.force


def test_tracking_cli_accepts_botsort_and_jsonl(tmp_path: Path) -> None:
    args = _script().build_parser().parse_args(
        [
            "--source", str(tmp_path / "input.mp4"),
            "--output", str(tmp_path / "output.mp4"),
            "--tracker", "botsort",
            "--tracker-config", str(tmp_path / "botsort.yaml"),
            "--jsonl",
        ]
    )

    assert args.tracker == "botsort"
    assert args.tracker_config == tmp_path / "botsort.yaml"
    assert args.jsonl is True


def test_tracking_cli_initializes_model_once(monkeypatch, tmp_path: Path) -> None:
    script = _script()
    constructions = []

    def detector_factory(**kwargs):
        constructions.append(kwargs)
        return object()

    def fake_run(detector, source, output, **kwargs):
        return {
            "total_frames": 1,
            "detection_frames": 1,
            "cumulative_created_track_ids": 0,
            "end_to_end_throughput_fps": 1.0,
            "output_path": str(output),
            "report_path": str(Path(output).with_suffix(".json")),
            "jsonl_path": None,
        }

    monkeypatch.setattr(script, "OpenCVDetector", detector_factory)
    monkeypatch.setattr(script, "run_tracking_video", fake_run)

    exit_code = script.main(
        ["--source", str(tmp_path / "input.mp4"), "--output", str(tmp_path / "output.mp4")]
    )

    assert exit_code == 0
    assert len(constructions) == 1
