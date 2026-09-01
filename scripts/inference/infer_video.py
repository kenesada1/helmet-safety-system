#!/usr/bin/env python3
"""Run M5 OpenCV inference for a local video."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.inference.opencv import OpenCVDetector  # noqa: E402
from helmet_safety.inference.videos import run_video_inference  # noqa: E402


DEFAULT_WEIGHTS = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "m45_yolo11s_e75_960_001"
    / "weights"
    / "best.pt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M5 OpenCV video inference using the E4 YOLO11s best.pt by default"
    )
    parser.add_argument("--source", type=Path, required=True, help="Local input video")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4, .m4v, or .avi file")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--progress-interval", type=int, default=30)
    parser.add_argument("--show", action="store_true", help="Show an optional OpenCV preview; off by default")
    parser.add_argument("--force", action="store_true", help="Allow existing inference outputs to be overwritten")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        detector = OpenCVDetector(
            weights=args.weights,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
        )
        report = run_video_inference(
            detector,
            args.source,
            args.output,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            fallback_fps=args.fallback_fps,
            progress_interval=args.progress_interval,
            force=args.force,
            show=args.show,
        )
    except Exception as exc:
        logging.error("video inference failed: %s", exc)
        return 1
    summary_keys = (
        "total_frames",
        "processed_frames",
        "skipped_frames",
        "average_inference_seconds",
        "average_full_frame_processing_seconds",
        "processing_throughput_fps",
        "helmet_detections",
        "no_helmet_detections",
    )
    print(
        json.dumps(
            {key: report[key] for key in summary_keys},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

