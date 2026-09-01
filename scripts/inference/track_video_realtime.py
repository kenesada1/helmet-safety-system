#!/usr/bin/env python3
"""M6 real-time tracking on a local video or camera.

Pipeline: bounded async capture queue -> detection every ``--frame-stride``-th
frame on E4 YOLO11s + ByteTrack (PyTorch CUDA) -> constant-velocity
interpolation on in-between frames -> confirmed-track gating + per-track
temporal class voting -> OpenCV drawing + writing.

Acceptance targets:
- full-chain average throughput >= 25 FPS
- P95 full-frame latency <= 40 ms
- queue end-to-end latency P95 <= 200 ms
- recordable drop rate < 1%
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.inference.opencv import OpenCVDetector  # noqa: E402
from helmet_safety.tracking.realtime import run_realtime_tracking_video  # noqa: E402


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
        description="M6 real-time video tracking (async queue + frame stride + interpolation)"
    )
    parser.add_argument("--source", required=True, help="Local video path or camera index such as 0")
    parser.add_argument("--output", type=Path, required=True, help="Output .mp4, .m4v, or .avi file")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="0", help="Ultralytics device, for example 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.10, help="Detector floor; 0.10-0.25 only continues tracks")
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--fp16", action="store_true", help="Run detector inference in FP16 (Tensor Cores); weights stay the same")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort"), default="bytetrack")
    parser.add_argument("--tracker-config", type=Path, default=None)
    parser.add_argument("--frame-stride", type=int, default=2, help="Detect every N-th frame")
    parser.add_argument("--queue-size", type=int, default=16, help="Bounded async capture queue capacity")
    parser.add_argument("--confirm-hits", type=int, default=3, help="Hits required to confirm a track")
    parser.add_argument("--vote-window", type=int, default=5, help="Class voting window size")
    parser.add_argument("--lost-ttl-seconds", type=float, default=1.5, help="Track retention TTL")
    parser.add_argument("--predict-gap-limit", type=int, default=4, help="Max frames without detection before interpolation stops")
    parser.add_argument("--id-switch-iou", type=float, default=0.5, help="IoU threshold for id-switch estimate")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--progress-interval", type=int, default=30)
    parser.add_argument("--no-throttle", action="store_true", help="Decode the file as fast as possible; drop old frames when the pipeline is slower")
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
            fp16=args.fp16,
        )
        report = run_realtime_tracking_video(
            detector,
            args.source,
            args.output,
            tracker_type=args.tracker,
            tracker_config=args.tracker_config,
            frame_stride=args.frame_stride,
            queue_maxsize=args.queue_size,
            confirm_hits=args.confirm_hits,
            vote_window=args.vote_window,
            lost_ttl_seconds=args.lost_ttl_seconds,
            predict_gap_limit_frames=args.predict_gap_limit,
            id_switch_iou=args.id_switch_iou,
            max_frames=args.max_frames,
            fallback_fps=args.fallback_fps,
            progress_interval=args.progress_interval,
            write_jsonl=args.jsonl,
            force=args.force,
            show=args.show,
            throttle=not args.no_throttle,
        )
    except Exception as exc:
        logging.error("realtime video tracking failed: %s", exc)
        return 1
    q = report["queue"]
    latency = report["latency_ms"]
    throughput = report["throughput"]
    quality = report["tracking_quality"]
    print(
        json.dumps(
            {
                "total_frames": report["total_frames"],
                "detection_frames": report["detection_frames"],
                "interpolated_frames": report["interpolated_frames"],
                "throughput_fps": throughput["end_to_end_throughput_fps"],
                "full_frame_p95_ms": latency["full_frame"]["p95"],
                "queue_e2e_p95_ms": latency["queue_e2e"]["p95"],
                "dropped_rate": q["dropped_rate"],
                "created_tracks": quality["created_tracks"],
                "confirmed_tracks": quality["confirmed_tracks"],
                "short_track_ratio_detection_le2": quality["short_track_ratio_detection_hits_le2"],
                "short_track_ratio_observed_le2": quality["short_track_ratio_observed_frames_le2"],
                "class_switches": quality["class_switches"],
                "recovered_tracks": quality["recovered_tracks"],
                "id_switch_estimate": quality["id_switch_estimate"],
                "output_path": report["output_path"],
                "report_path": report["report_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
