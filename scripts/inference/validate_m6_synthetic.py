#!/usr/bin/env python3
"""Create a deterministic moving-box video and compare M6 trackers without model/GPU use."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.inference.opencv import Detection, InferenceResult  # noqa: E402
from helmet_safety.tracking.video import run_tracking_video  # noqa: E402


FPS = 15.0
FRAME_COUNT = 90
SIZE = (320, 240)


def ground_truth(frame_index: int) -> list[tuple[str, Detection]]:
    items: list[tuple[str, Detection]] = []
    if 5 <= frame_index < 86 and frame_index not in range(32, 40):
        x = 10 + 3 * (frame_index - 5)
        class_id = 1 if 55 <= frame_index < 60 else 0
        confidence = 0.18 if frame_index in {40, 41} else 0.90
        items.append(
            (
                "a",
                Detection(class_id, ("helmet", "no_helmet")[class_id], confidence, (x, 74, x + 24, 116)),
            )
        )
    if 5 <= frame_index < 86:
        x = 270 - 3 * (frame_index - 5)
        items.append(("b", Detection(1, "no_helmet", 0.88, (x, 92, x + 24, 134))))
    if 45 <= frame_index < 70:
        y = 205 - 3 * (frame_index - 45)
        items.append(("c", Detection(0, "helmet", 0.82, (220, y, 246, y + 32))))
    if 20 <= frame_index < 23:
        items.append(("short", Detection(1, "no_helmet", 0.76, (42, 172, 61, 203))))
    return items


class SyntheticDetector:
    """Deterministic injected detector; it deliberately never loads a model."""

    def __init__(self) -> None:
        self.frame_index = 0
        self.weights = "synthetic-ground-truth (no model loaded)"
        self.device = "cpu"
        self.imgsz = 960
        self.conf = 0.10
        self.iou = 0.70
        self.max_det = 300

    def predict_bgr(self, image: np.ndarray) -> InferenceResult:
        detections = [detection for _, detection in ground_truth(self.frame_index)]
        self.frame_index += 1
        return InferenceResult(detections=detections, inference_seconds=0.0)


def write_source(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError(f"unable to create synthetic source: {path}")
    try:
        for frame_index in range(FRAME_COUNT):
            frame = np.full((SIZE[1], SIZE[0], 3), 28, dtype=np.uint8)
            cv2.putText(
                frame,
                f"synthetic continuous motion | frame {frame_index}",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            for key, detection in ground_truth(frame_index):
                x1, y1, x2, y2 = (int(value) for value in detection.xyxy)
                color = (0, 180, 0) if detection.class_id == 0 else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
                cv2.putText(frame, key, (x1, max(35, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            writer.write(frame)
    finally:
        writer.release()


def iou(left: list[float] | tuple[float, ...], right: tuple[float, ...]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def analyze(jsonl_path: Path, report: dict[str, object]) -> dict[str, object]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    id_sequences: dict[str, list[tuple[int, int]]] = {key: [] for key in ("a", "b", "c", "short")}
    unmatched_visible_frames = 0
    for row in rows:
        frame_index = int(row["frame_index"])
        active = [item for item in row["observations"] if item["track_state"] == "active"]
        used: set[int] = set()
        for key, detection in ground_truth(frame_index):
            candidates = [
                (iou(item["xyxy"], detection.xyxy), index, item)
                for index, item in enumerate(active)
                if index not in used
            ]
            score, index, matched = max(candidates, default=(0.0, -1, None), key=lambda value: value[0])
            if matched is None or score < 0.20:
                unmatched_visible_frames += 1
                continue
            used.add(index)
            id_sequences[key].append((frame_index, int(matched["track_id"])))
    switches = 0
    fragments: dict[str, list[int]] = {}
    for key, sequence in id_sequences.items():
        ids = [track_id for _, track_id in sequence]
        switches += sum(previous != current for previous, current in zip(ids, ids[1:]))
        fragments[key] = sorted(set(ids))
    a_before = next((track_id for frame, track_id in reversed(id_sequences["a"]) if frame < 32), None)
    a_after = next((track_id for frame, track_id in id_sequences["a"] if frame >= 40), None)
    summaries = report["track_summaries"]
    return {
        "id_switches": switches,
        "track_fragments_by_object": fragments,
        "unmatched_visible_frames": unmatched_visible_frames,
        "occlusion_recovery": {
            "object": "a",
            "last_id_before_frames_32_to_39": a_before,
            "first_id_after_frame_40": a_after,
            "recovered_same_id": a_before is not None and a_before == a_after,
        },
        "short_lived_track_ids_hits_le_3": [
            item["track_id"] for item in summaries if int(item["hits"]) <= 3
        ],
        "created_track_ids": report["cumulative_created_track_ids"],
        "track_transitions": report["track_transitions"],
        "end_to_end_throughput_fps": report["end_to_end_throughput_fps"],
        "average_tracking_seconds": report["average_tracking_seconds"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible M6 synthetic-motion validation")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "tracking" / "m6_smoke",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = args.output_directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "synthetic_continuous_motion.mp4"
    targets = [source, directory / "synthetic_validation.json"]
    for tracker_type in ("bytetrack", "botsort"):
        targets.extend(
            directory / f"synthetic_{tracker_type}{suffix}"
            for suffix in (".mp4", ".json", ".jsonl")
        )
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"synthetic validation output exists: {existing[0]}; pass --force")
    write_source(source)
    comparison: dict[str, object] = {
        "schema_version": "m6-synthetic-validation-v1",
        "source": str(source),
        "scenario": {
            "frames": FRAME_COUNT,
            "fps": FPS,
            "features": [
                "two simultaneous targets moving in opposite directions",
                "short occlusion on frames 32-39",
                "low-confidence recovery on frames 40-41",
                "class fluctuation on frames 55-59",
                "third target entering and leaving",
                "one deliberately short-lived target",
            ],
        },
        "results": {},
        "limitation": (
            "This is deterministic synthetic motion with injected detections. It validates tracker state and video "
            "plumbing, but it is not a substitute for real continuous-person ID Switch acceptance."
        ),
    }
    for tracker_type in ("bytetrack", "botsort"):
        output = directory / f"synthetic_{tracker_type}.mp4"
        report = run_tracking_video(
            SyntheticDetector(),
            source,
            output,
            tracker_type=tracker_type,
            frame_stride=1,
            lost_ttl_seconds=1.0,
            write_jsonl=True,
            force=args.force,
            show=False,
        )
        comparison["results"][tracker_type] = analyze(output.with_suffix(".jsonl"), report)
    comparison_path = directory / "synthetic_validation.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
