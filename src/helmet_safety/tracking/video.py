"""OpenCV video I/O and reporting around the reusable M6 tracker."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2

from helmet_safety.inference.opencv import OpenCVDetector
from helmet_safety.inference.videos import DEFAULT_FALLBACK_FPS, SUPPORTED_OUTPUT_EXTENSIONS, _open_writer

from .core import DEFAULT_CONFIG_PATHS, TrackingSession, calculate_track_buffer, load_tracker_config
from .drawing import draw_tracks


LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_tracking_video(
    detector: OpenCVDetector,
    source: Path | str | int,
    output: Path | str,
    *,
    tracker_type: str = "bytetrack",
    tracker_config: Path | str | None = None,
    frame_stride: int = 1,
    max_frames: int | None = None,
    lost_ttl_seconds: float = 1.0,
    fallback_fps: float = DEFAULT_FALLBACK_FPS,
    progress_interval: int = 30,
    write_jsonl: bool = False,
    force: bool = False,
    show: bool = False,
    capture_factory: Callable[[object], object] = cv2.VideoCapture,
    writer_factory: Callable[..., object] = cv2.VideoWriter,
    tracker_factory: Callable[..., TrackingSession] = TrackingSession,
) -> dict[str, object]:
    source_text = str(source)
    source_candidate = Path(source_text).expanduser()
    is_camera = isinstance(source, int) or (
        isinstance(source, str)
        and source.strip().isdigit()
        and not source_candidate.exists()
    )
    camera_index = int(source) if is_camera else None
    input_path = None if is_camera else source_candidate.resolve()
    input_label = f"camera:{camera_index}" if is_camera else str(input_path)
    capture_source: object = camera_index if is_camera else str(input_path)
    output_path = Path(output).expanduser().resolve()
    report_path = output_path.with_suffix(".json")
    jsonl_path = output_path.with_suffix(".jsonl")
    if input_path is not None and not input_path.is_file():
        raise FileNotFoundError(f"video input does not exist: {input_path}")
    if input_path is not None and input_path == output_path:
        raise ValueError(f"output video would overwrite input video: {input_path}")
    if output_path.suffix.lower() not in SUPPORTED_OUTPUT_EXTENSIONS:
        raise ValueError("output video must use .mp4, .m4v, or .avi")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided")
    if not math.isfinite(fallback_fps) or fallback_fps <= 0:
        raise ValueError("fallback_fps must be a finite positive number")
    if not math.isfinite(lost_ttl_seconds) or lost_ttl_seconds <= 0:
        raise ValueError("lost_ttl_seconds must be a finite positive number")
    if progress_interval < 1:
        raise ValueError("progress_interval must be positive")
    targets = [output_path, report_path]
    if write_jsonl:
        targets.append(jsonl_path)
    for target in targets:
        if target.exists() and not force:
            raise FileExistsError(f"output already exists: {target}; pass --force to overwrite")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = capture_factory(capture_source)
    writer: object | None = None
    jsonl_file = None
    released = {
        "capture_released": False,
        "writer_released": False,
        "jsonl_closed": not write_jsonl,
        "windows_destroyed": not show,
    }
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"unable to open input video with OpenCV: {input_label}")
    raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fallback_used = not math.isfinite(raw_fps) or raw_fps <= 0
    source_fps = fallback_fps if fallback_used else raw_fps
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width < 1 or height < 1:
        capture.release()
        raise ValueError(f"input video reports invalid dimensions {width}x{height}: {input_label}")

    try:
        track_buffer = calculate_track_buffer(source_fps, frame_stride, lost_ttl_seconds)
        config = load_tracker_config(
            tracker_type,
            track_buffer=track_buffer,
            config_path=tracker_config,
            device=str(getattr(detector, "device", "cpu")),
        )
        session = tracker_factory(tracker_type=tracker_type, config=config)
        session.reset()
    except Exception:
        capture.release()
        released["capture_released"] = True
        raise

    total_frames = 0
    detection_frames = 0
    total_inference = 0.0
    total_tracking = 0.0
    total_drawing = 0.0
    total_frame_time = 0.0
    active_sum = 0
    lost_sum = 0
    class_counts = {"helmet": 0, "no_helmet": 0}
    class_detection_counts = {"helmet": 0, "no_helmet": 0}
    prediction_count = 0
    pipeline_started = perf_counter()
    codec = ""
    try:
        writer, codec = _open_writer(output_path, source_fps, (width, height), writer_factory)
        if write_jsonl:
            jsonl_file = jsonl_path.open("w", encoding="utf-8")
        while max_frames is None or total_frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"OpenCV returned an invalid frame at index {total_frames}")
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"video frame dimensions changed from {width}x{height} to "
                    f"{frame.shape[1]}x{frame.shape[0]} at index {total_frames}"
                )
            frame_started = perf_counter()
            observations = []
            detection_performed = total_frames % frame_stride == 0
            if detection_performed:
                inference = detector.predict_bgr(frame)
                total_inference += float(inference.inference_seconds)
                tracked = session.update(
                    inference.detections,
                    frame,
                    frame_index=total_frames,
                    timestamp_seconds=total_frames / source_fps,
                )
                total_tracking += tracked.tracking_seconds
                observations = tracked.observations
                detection_frames += 1
                for observation in observations:
                    class_counts[observation.class_name] += 1
                    if observation.is_prediction:
                        prediction_count += 1
                    else:
                        class_detection_counts[observation.class_name] += 1
            elapsed_before_draw = perf_counter() - frame_started
            processing_fps = 1.0 / elapsed_before_draw if elapsed_before_draw > 0 else 0.0
            drawing_started = perf_counter()
            output_frame = draw_tracks(
                frame,
                observations,
                active_tracks=session.current_active_count,
                lost_tracks=session.current_lost_count,
                created_tracks=session.statistics["created_tracks"],
                processing_fps=processing_fps,
                tracker_type=tracker_type,
            )
            total_drawing += perf_counter() - drawing_started
            writer.write(output_frame)
            active_sum += session.current_active_count
            lost_sum += session.current_lost_count
            if jsonl_file is not None:
                row = {
                    "frame_index": total_frames,
                    "timestamp_seconds": total_frames / source_fps,
                    "detection_performed": detection_performed,
                    "observations": [item.to_dict() for item in observations],
                }
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_frames += 1
            total_frame_time += perf_counter() - frame_started
            if total_frames % progress_interval == 0:
                LOGGER.info(
                    "tracking progress: %d frames written, %d detection frames",
                    total_frames,
                    detection_frames,
                )
            if show:
                cv2.imshow("helmet-safety M6 tracking", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        released["capture_released"] = True
        if writer is not None:
            writer.release()
        released["writer_released"] = True
        if jsonl_file is not None:
            jsonl_file.close()
        released["jsonl_closed"] = True
        if show:
            cv2.destroyAllWindows()
        released["windows_destroyed"] = True
    pipeline_seconds = perf_counter() - pipeline_started
    if total_frames == 0:
        raise ValueError(f"input video contains no readable frames: {input_label}")

    report: dict[str, object] = {
        "schema_version": "m6-tracking-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": input_label,
        "output_path": str(output_path),
        "report_path": str(report_path),
        "jsonl_path": str(jsonl_path) if write_jsonl else None,
        "model": {
            "weights": str(getattr(detector, "weights", "unknown")),
            "device": str(getattr(detector, "device", "unknown")),
            "imgsz": int(getattr(detector, "imgsz", 0)),
            "conf": float(getattr(detector, "conf", 0.0)),
            "iou": float(getattr(detector, "iou", 0.0)),
            "max_det": int(getattr(detector, "max_det", 0)),
        },
        "tracker_type": tracker_type,
        "tracker_config_path": (
            str(Path(tracker_config).resolve())
            if tracker_config
            else str(DEFAULT_CONFIG_PATHS[tracker_type].resolve())
        ),
        "tracker_config": config.to_dict(),
        "width": width,
        "height": height,
        "codec": codec,
        "declared_source_frames": max(0, declared_frames),
        "total_frames": total_frames,
        "detection_frames": detection_frames,
        "skipped_frames": total_frames - detection_frames,
        "dropped_frames": 0,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "raw_source_fps": raw_fps if math.isfinite(raw_fps) else None,
        "source_fps": source_fps,
        "detection_fps": source_fps / frame_stride,
        "output_fps": source_fps,
        "fps_fallback_used": fallback_used,
        "fallback_fps": fallback_fps if fallback_used else None,
        "lost_ttl_seconds": lost_ttl_seconds,
        "track_buffer": track_buffer,
        "track_buffer_seconds": track_buffer * frame_stride / source_fps,
        "cumulative_created_track_ids": session.statistics["created_tracks"],
        "average_active_tracks": active_sum / total_frames,
        "average_lost_tracks": lost_sum / total_frames,
        "track_transitions": dict(session.statistics),
        "class_observation_boxes": class_counts,
        "class_detection_observation_boxes": class_detection_counts,
        "prediction_observation_boxes": prediction_count,
        "track_summaries": session.track_summaries(),
        "average_inference_seconds": total_inference / detection_frames if detection_frames else 0.0,
        "average_tracking_seconds": total_tracking / detection_frames if detection_frames else 0.0,
        "average_drawing_seconds": total_drawing / total_frames,
        "average_full_frame_seconds": total_frame_time / total_frames,
        "pipeline_seconds": pipeline_seconds,
        "end_to_end_throughput_fps": total_frames / pipeline_seconds if pipeline_seconds > 0 else 0.0,
        "resource_release_status": released,
        "note": (
            "This report contains tracks and per-frame observation boxes only. "
            "They are not unique safety-violation events; event deduplication belongs to M7. "
            "Detections from 0.10 to below 0.25 may continue existing tracks, while new tracks require 0.25. "
            "Skipped frames contain no reused boxes."
        ),
    }
    _write_json(report_path, report)
    return report
