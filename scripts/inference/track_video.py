#!/usr/bin/env python3
"""Run M6 ByteTrack or BoT-SORT tracking on a local video."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.inference.opencv import OpenCVDetector  # noqa: E402
from helmet_safety.tracking.video import run_tracking_video  # noqa: E402


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
        description="M6 video tracking with E4 YOLO11s and ByteTrack by default"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Local input video path or non-negative camera index such as 0",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4, .m4v, or .avi file")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.10, help="Detector floor; 0.10-0.25 only continues tracks")
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--tracker", choices=("bytetrack", "botsort"), default="bytetrack")
    parser.add_argument("--tracker-config", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--lost-ttl-seconds", type=float, default=1.0)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--progress-interval", type=int, default=30)
    parser.add_argument("--jsonl", action="store_true", help="Write one explicit tracking record per source frame")
    parser.add_argument("--show", action="store_true", help="Show an optional OpenCV preview; off by default")
    parser.add_argument("--force", action="store_true", help="Allow output video/JSON/JSONL overwrite")
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
        report = run_tracking_video(
            detector,
            args.source,
            args.output,
            tracker_type=args.tracker,
            tracker_config=args.tracker_config,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            lost_ttl_seconds=args.lost_ttl_seconds,
            fallback_fps=args.fallback_fps,
            progress_interval=args.progress_interval,
            write_jsonl=args.jsonl,
            force=args.force,
            show=args.show,
        )
    except Exception as exc:
        logging.error("video tracking failed: %s", exc)
        return 1
    print(
        json.dumps(
            {
                "total_frames": report["total_frames"],
                "detection_frames": report["detection_frames"],
                "cumulative_created_track_ids": report["cumulative_created_track_ids"],
                "end_to_end_throughput_fps": report["end_to_end_throughput_fps"],
                "output_path": report["output_path"],
                "report_path": report["report_path"],
                "jsonl_path": report["jsonl_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
