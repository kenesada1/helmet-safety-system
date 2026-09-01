"""OpenCV VideoCapture/VideoWriter inference flow for M5."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Callable

import cv2

from .opencv import OpenCVDetector, draw_detections


LOGGER = logging.getLogger(__name__)
DEFAULT_FALLBACK_FPS = 30.0
SUPPORTED_OUTPUT_EXTENSIONS = frozenset({".mp4", ".m4v", ".avi"})


def _writer_codecs(suffix: str) -> tuple[str, ...]:
    return ("mp4v", "avc1") if suffix in {".mp4", ".m4v"} else ("XVID", "MJPG")


def _open_writer(
    output: Path,
    fps: float,
    size: tuple[int, int],
    writer_factory: Callable[..., object],
    codecs: tuple[str, ...] | None = None,
) -> tuple[object, str]:
    errors: list[str] = []
    ordered = codecs if codecs is not None else _writer_codecs(output.suffix.lower())
    for codec in ordered:
        writer = writer_factory(
            str(output), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
        errors.append(codec)
    raise RuntimeError(
        f"unable to create output video {output} with codecs {', '.join(errors)}"
    )


def _write_json(path: Path, report: object) -> None:
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run_video_inference(
    detector: OpenCVDetector,
    source: Path | str,
    output: Path | str,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    fallback_fps: float = DEFAULT_FALLBACK_FPS,
    progress_interval: int = 30,
    force: bool = False,
    show: bool = False,
    capture_factory: Callable[[str], object] = cv2.VideoCapture,
    writer_factory: Callable[..., object] = cv2.VideoWriter,
) -> dict[str, object]:
    input_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    report_path = output_path.with_suffix(".json")
    if not input_path.is_file():
        raise FileNotFoundError(f"video input does not exist: {input_path}")
    if output_path == input_path:
        raise ValueError(f"output video would overwrite input video: {input_path}")
    if output_path.suffix.lower() not in SUPPORTED_OUTPUT_EXTENSIONS:
        raise ValueError("output video must use .mp4, .m4v, or .avi")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided")
    if fallback_fps <= 0 or not math.isfinite(fallback_fps):
        raise ValueError("fallback_fps must be a finite positive number")
    if progress_interval < 1:
        raise ValueError("progress_interval must be positive")
    for target in (output_path, report_path):
        if target.exists() and not force:
            raise FileExistsError(
                f"output already exists: {target}; pass --force to overwrite"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = capture_factory(str(input_path))
    writer: object | None = None
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"unable to open input video with OpenCV: {input_path}")

    raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps_fallback_used = not math.isfinite(raw_fps) or raw_fps <= 0
    output_fps = fallback_fps if fps_fallback_used else raw_fps
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width < 1 or height < 1:
        capture.release()
        raise ValueError(
            f"input video reports invalid dimensions {width}x{height}: {input_path}"
        )

    total_frames = 0
    processed_frames = 0
    helmet_detections = 0
    no_helmet_detections = 0
    total_inference_seconds = 0.0
    total_frame_processing_seconds = 0.0
    pipeline_started = perf_counter()
    codec = ""
    try:
        writer, codec = _open_writer(
            output_path, output_fps, (width, height), writer_factory
        )
        while max_frames is None or total_frames < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(f"OpenCV returned an invalid frame at index {total_frames}")
            if frame.shape[1] != width or frame.shape[0] != height:
                raise ValueError(
                    "video frame dimensions changed from "
                    f"{width}x{height} to {frame.shape[1]}x{frame.shape[0]} "
                    f"at index {total_frames}"
                )
            frame_started = perf_counter()
            output_frame = frame
            if total_frames % frame_stride == 0:
                result = detector.predict_bgr(frame)
                current_elapsed = perf_counter() - frame_started
                current_fps = 1.0 / current_elapsed if current_elapsed > 0 else 0.0
                output_frame, counts = draw_detections(
                    frame, result.detections, processing_fps=current_fps
                )
                processed_frames += 1
                total_inference_seconds += result.inference_seconds
                helmet_detections += counts["helmet"]
                no_helmet_detections += counts["no_helmet"]
            writer.write(output_frame)
            total_frames += 1
            total_frame_processing_seconds += perf_counter() - frame_started
            if total_frames % progress_interval == 0:
                LOGGER.info(
                    "video progress: %d frames written, %d frames inferred",
                    total_frames,
                    processed_frames,
                )
            if show:
                cv2.imshow("helmet-safety M5", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    pipeline_seconds = perf_counter() - pipeline_started

    if total_frames == 0:
        raise ValueError(f"input video contains no readable frames: {input_path}")
    skipped_frames = total_frames - processed_frames
    report: dict[str, object] = {
        "schema_version": "m5-video-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "report_path": str(report_path),
        "width": width,
        "height": height,
        "declared_source_frames": max(0, declared_frames),
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "skipped_frames": skipped_frames,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "source_fps": raw_fps if math.isfinite(raw_fps) else None,
        "output_fps": output_fps,
        "fps_fallback_used": fps_fallback_used,
        "fallback_fps": fallback_fps if fps_fallback_used else None,
        "codec": codec,
        "average_inference_seconds": (
            total_inference_seconds / processed_frames if processed_frames else 0.0
        ),
        "average_full_frame_processing_seconds": (
            total_frame_processing_seconds / total_frames
        ),
        "processing_throughput_fps": (
            processed_frames / pipeline_seconds if pipeline_seconds > 0 else 0.0
        ),
        "output_frame_throughput_fps": (
            total_frames / pipeline_seconds if pipeline_seconds > 0 else 0.0
        ),
        "pipeline_seconds": pipeline_seconds,
        "helmet_detections": helmet_detections,
        "no_helmet_detections": no_helmet_detections,
        "total_detections": helmet_detections + no_helmet_detections,
        "note": (
            "Counts are per-frame detection boxes, not unique people or violation events. "
            "Skipped frames are written without detections."
        ),
    }
    _write_json(report_path, report)
    return report

