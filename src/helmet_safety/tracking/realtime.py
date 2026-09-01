"""Real-time tracking pipeline for M6.

Design goals (M6 real-time acceptance):
- Full-chain average throughput >= 25 FPS.
- P95 full-frame latency <= 40 ms.
- Bounded async capture queue with recordable drop rate < 1%.
- Detection on every ``stride``-th frame plus interpolated prediction on the
  in-between frames so tracks stay visually continuous.
- Confirmed-track gating plus per-track temporal class voting to reduce
  short-lived (hits<=2) tracks, class switches, and ID fragmentation.

This module is additive: it reuses :class:`TrackingSession` from ``core`` and
:class:`OpenCVDetector` from ``inference.opencv`` without modifying them.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
from threading import Condition, Thread
from time import perf_counter, sleep
from typing import Callable, Sequence

import cv2
import numpy as np

from helmet_safety.inference.opencv import CLASS_NAMES, Detection, OpenCVDetector
from helmet_safety.inference.videos import (
    DEFAULT_FALLBACK_FPS,
    SUPPORTED_OUTPUT_EXTENSIONS,
    _open_writer,
)
from helmet_safety.tracking.drawing import draw_tracks

from .core import (
    FrameTrackingResult,
    TrackObservation,
    TrackingSession,
    calculate_track_buffer,
    load_tracker_config,
)

__all__ = [
    "BoundedFrameQueue",
    "TemporalClassVoter",
    "TrackMotionBuffer",
    "RealtimeTrackingSession",
    "run_realtime_tracking_video",
]

LOGGER = logging.getLogger(__name__)

#: Queue item layout: (frame_index, timestamp_seconds, frame, enqueue_wall_clock).
_QueueItem = tuple[int, float, np.ndarray, float]


def _iou_xyxy(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _percentiles(values: Sequence[float], points: Sequence[int]) -> dict[str, float]:
    """Nearest-rank percentiles. Empty input yields 0.0 for every point."""
    if not values:
        return {f"p{point}": 0.0 for point in points}
    ordered = sorted(float(value) for value in values)
    result: dict[str, float] = {}
    for point in points:
        index = max(0, int(math.ceil(point / 100.0 * len(ordered))) - 1)
        result[f"p{point}"] = ordered[index]
    return result


class BoundedFrameQueue:
    """Thread-safe bounded queue of source frames.

    The capture thread calls :meth:`put`; the processing thread calls
    :meth:`get`. When the queue is full a new frame evicts the *oldest* frame
    (keeping the freshest data for a live stream) and increments ``dropped`` so
    the real drop rate is measurable.
    """

    def __init__(self, maxsize: int = 4) -> None:
        if maxsize < 1:
            raise ValueError("queue maxsize must be positive")
        self.maxsize = int(maxsize)
        self._buffer: deque[_QueueItem] = deque()
        self._cv = Condition()
        self._stopped = False
        self.enqueued = 0
        self.dropped = 0
        self.dequeued = 0

    def put(self, item: _QueueItem) -> bool:
        """Return False when the queue has been stopped (capture should exit)."""
        with self._cv:
            if self._stopped:
                return False
            self.enqueued += 1
            if len(self._buffer) >= self.maxsize:
                self._buffer.popleft()
                self.dropped += 1
            self._buffer.append(item)
            self._cv.notify()
        return True

    def get(self) -> _QueueItem | None:
        """Block until a frame is available or the queue is stopped+empty."""
        with self._cv:
            while not self._buffer and not self._stopped:
                self._cv.wait()
            if not self._buffer:
                return None
            self.dequeued += 1
            return self._buffer.popleft()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()

    def __len__(self) -> int:
        with self._cv:
            return len(self._buffer)


class TemporalClassVoter:
    """Per-track temporal majority vote for the stable class label.

    A class only becomes the stable class once it appears ``min_majority``
    times inside the sliding window. Every change of the stable class counts as
    one class switch, which feeds the class-switch report signal.
    """

    def __init__(self, window: int = 5, min_majority: int = 3) -> None:
        if window < 1 or min_majority < 1:
            raise ValueError("window and min_majority must be positive")
        if min_majority > window:
            raise ValueError("min_majority cannot exceed window")
        self.window = int(window)
        self.min_majority = int(min_majority)
        self._recent: deque[int] = deque(maxlen=self.window)
        self.stable_class: int | None = None
        self.switches = 0

    def observe(self, class_id: int) -> bool:
        """Feed one class observation; return True when the stable class flipped."""
        self._recent.append(class_id)
        if len(self._recent) >= self.min_majority:
            candidate, count = Counter(self._recent).most_common(1)[0]
            if count >= self.min_majority and candidate != self.stable_class:
                self.stable_class = candidate
                self.switches += 1
                return True
        return False

    def seen_classes(self) -> list[int]:
        return list(self._recent)


class TrackMotionBuffer:
    """Short per-track position history used to extrapolate in-between frames.

    Uses a constant-velocity model on the last two detection updates. With a
    single point it holds the box still. ``predict`` returns raw corner values
    (no clamping) so the drawing layer can clip them.
    """

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 2:
            raise ValueError("motion buffer capacity must be at least 2")
        self._points: deque[tuple[int, np.ndarray]] = deque(maxlen=int(capacity))

    def update(self, frame_index: int, xyxy: Sequence[float]) -> None:
        self._points.append((int(frame_index), np.asarray(xyxy, dtype=np.float32).reshape(4)))

    def predict(self, target_frame_index: int) -> tuple[float, float, float, float] | None:
        count = len(self._points)
        if count == 0:
            return None
        last_frame, last_box = self._points[-1]
        if count == 1:
            return tuple(float(value) for value in last_box)  # type: ignore[return-value]
        prev_frame, prev_box = self._points[-2]
        delta = last_frame - prev_frame
        if delta <= 0:
            return tuple(float(value) for value in last_box)  # type: ignore[return-value]
        steps = int(target_frame_index) - last_frame
        if steps < 0:
            return tuple(float(value) for value in last_box)  # type: ignore[return-value]
        velocity = (last_box - prev_box) / delta
        predicted = last_box + velocity * steps
        return tuple(float(value) for value in predicted)


@dataclass(slots=True)
class _RealtimeTrack:
    track_id: int
    hits: int = 0
    observed_frames: int = 0
    first_observed_frame: int = -1
    last_detection_frame: int = -1
    last_confidence: float = 0.0
    current_class_id: int = 0
    last_xyxy: tuple[float, float, float, float] | None = None
    is_active: bool = False
    seen_classes: set[int] = field(default_factory=set)
    motion: TrackMotionBuffer = field(default_factory=TrackMotionBuffer)
    voter: TemporalClassVoter = field(default_factory=TemporalClassVoter)

    @property
    def stable_class_id(self) -> int | None:
        return self.voter.stable_class


class RealtimeTrackingSession:
    """Tracks + per-track interpolation/voting/confirmation on top of core.

    Detection frames call :meth:`update_detection`, which forwards to the
    underlying :class:`TrackingSession`. In-between frames call
    :meth:`predict_interpolated`, which extrapolates the active tracks and emits
    ``is_prediction=True`` observations without touching the tracker state, so
    the Kalman association and hit counters are never polluted by synthetic
    boxes.
    """

    def __init__(
        self,
        session: TrackingSession,
        *,
        frame_stride: int = 2,
        confirm_hits: int = 3,
        vote_window: int = 5,
        vote_min_majority: int = 3,
        predict_gap_limit_frames: int = 4,
        id_switch_iou: float = 0.5,
    ) -> None:
        if frame_stride < 1 or confirm_hits < 1:
            raise ValueError("frame_stride and confirm_hits must be positive")
        self._session = session
        self.frame_stride = int(frame_stride)
        self.confirm_hits = int(confirm_hits)
        self.vote_window = int(vote_window)
        self.vote_min_majority = int(vote_min_majority)
        self.predict_gap_limit_frames = int(predict_gap_limit_frames)
        self.id_switch_iou = float(id_switch_iou)
        self.tracks: dict[int, _RealtimeTrack] = {}
        self._lost_pool: list[tuple[int, tuple[float, float, float, float], int]] = []
        self.id_switch_estimate = 0

    @property
    def statistics(self) -> dict[str, int]:
        return self._session.statistics

    @property
    def current_active_count(self) -> int:
        return self._session.current_active_count

    @property
    def current_lost_count(self) -> int:
        return self._session.current_lost_count

    def update_detection(
        self,
        detections: Sequence[Detection],
        image: np.ndarray,
        *,
        frame_index: int,
        timestamp_seconds: float,
    ) -> FrameTrackingResult:
        result = self._session.update(
            detections, image, frame_index=frame_index, timestamp_seconds=timestamp_seconds
        )

        active_now: set[int] = set()
        lost_now: list[tuple[int, tuple[float, float, float, float]]] = []
        for observation in result.observations:
            track_id = observation.track_id
            rt = self.tracks.get(track_id)
            if rt is None:
                rt = _RealtimeTrack(track_id=track_id, first_observed_frame=frame_index)
                self.tracks[track_id] = rt
                if frame_index > 0:
                    for lost_id, lost_box, lost_at in self._lost_pool:
                        if (
                            frame_index - lost_at <= self.predict_gap_limit_frames
                            and _iou_xyxy(observation.xyxy, lost_box) >= self.id_switch_iou
                        ):
                            self.id_switch_estimate += 1
                            break
            if observation.track_state == "active":
                active_now.add(track_id)
                rt.hits = observation.hits
                rt.last_detection_frame = frame_index
                rt.last_confidence = observation.confidence
                rt.current_class_id = observation.class_id
                rt.last_xyxy = observation.xyxy
                rt.is_active = True
                rt.seen_classes.add(observation.class_id)
                rt.motion.update(frame_index, observation.xyxy)
                rt.voter.observe(observation.class_id)
                rt.observed_frames += 1
            elif observation.track_state == "lost":
                lost_now.append((track_id, observation.xyxy))
                rt.is_active = False
                rt.observed_frames += 1

        kept: list[tuple[int, tuple[float, float, float, float], int]] = []
        for lost_id, lost_box, lost_at in self._lost_pool:
            if lost_id in active_now:
                continue
            if frame_index - lost_at > self.predict_gap_limit_frames:
                continue
            kept.append((lost_id, lost_box, lost_at))
        for track_id, box in lost_now:
            kept.append((track_id, box, frame_index))
        self._lost_pool = kept

        # Tracks that did not appear in this detection frame (neither active nor
        # lost) are not interpolated on the next in-between frame, so the set of
        # boxes drawn on the interpolated frame exactly matches the detection
        # frame. This prevents flicker where a track's box alternates on/off.
        appeared = active_now | {track_id for track_id, _ in lost_now}
        for track_id, rt in self.tracks.items():
            if track_id not in appeared:
                rt.is_active = False
        return result

    def predict_interpolated(
        self, frame_index: int, timestamp_seconds: float
    ) -> list[TrackObservation]:
        """Extrapolate the active tracks for an in-between (non-detection) frame."""
        observations: list[TrackObservation] = []
        for track_id, rt in self.tracks.items():
            if not rt.is_active:
                continue
            if rt.last_detection_frame < 0:
                continue
            if frame_index - rt.last_detection_frame > self.predict_gap_limit_frames:
                continue
            predicted = rt.motion.predict(frame_index)
            if predicted is None:
                continue
            # Use the tracker's current class so the box color matches the
            # detection frames (no color flicker); the voted stable class is
            # still reported in finalize().
            class_id = rt.current_class_id
            class_name = CLASS_NAMES.get(class_id, str(class_id))
            track_age = int(frame_index) - rt.first_observed_frame + 1
            observations.append(
                TrackObservation(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=rt.last_confidence,
                    xyxy=predicted,
                    frame_index=frame_index,
                    timestamp_seconds=float(timestamp_seconds),
                    track_state="active",
                    track_age=track_age,
                    hits=rt.hits,
                    time_since_update=frame_index - rt.last_detection_frame,
                    is_prediction=True,
                )
            )
            rt.observed_frames += 1
        return observations

    def finalize(self) -> dict[str, object]:
        """Aggregate per-track reports once the stream is exhausted."""
        summaries: list[dict[str, object]] = []
        detection_short = 0
        observed_short = 0
        confirmed = 0
        mixed_class = 0
        for track_id, rt in sorted(self.tracks.items()):
            if rt.hits <= 2:
                detection_short += 1
            if rt.observed_frames <= 2:
                observed_short += 1
            if rt.hits >= self.confirm_hits:
                confirmed += 1
            if len(rt.seen_classes) > 1:
                mixed_class += 1
            summaries.append(
                {
                    "track_id": track_id,
                    "hits": rt.hits,
                    "observed_frames": rt.observed_frames,
                    "first_source_frame": rt.first_observed_frame,
                    "last_detection_frame": rt.last_detection_frame,
                    "stable_class_id": rt.stable_class_id,
                    "current_class_id": rt.current_class_id,
                    "class_switches": rt.voter.switches,
                    "seen_classes": sorted(rt.seen_classes),
                }
            )
        created = len(self.tracks)
        class_switches_total = sum(int(item["class_switches"]) for item in summaries)
        return {
            "created_tracks": created,
            "confirmed_tracks": confirmed,
            "tentative_tracks": created - confirmed,
            "class_switches": class_switches_total,
            "id_switch_estimate": self.id_switch_estimate,
            "recovered_tracks": self._session.statistics["recovered_tracks"],
            "removed_tracks": self._session.statistics["removed_tracks"],
            "lost_tracks": self._session.statistics["lost_tracks"],
            "short_tracks_detection_hits_le2": detection_short,
            "short_tracks_observed_frames_le2": observed_short,
            "short_track_ratio_detection_hits_le2": (
                detection_short / created if created else 0.0
            ),
            "short_track_ratio_observed_frames_le2": (
                observed_short / created if created else 0.0
            ),
            "mixed_class_tracks": mixed_class,
            "track_summaries": summaries,
        }


def _capture_loop(
    capture: object, queue: BoundedFrameQueue, source_fps: float, throttle: bool
) -> None:
    """Read frames into the queue, optionally pacing to the source frame rate.

    ``throttle=True`` emulates a live camera: frames are read at ``source_fps``
    so a pipeline faster than the source never overflows the queue and the drop
    rate stays ~0. ``throttle=False`` decodes the file as fast as possible and
    the bounded queue drops the oldest frames when the pipeline is slower.
    """
    index = 0
    started = perf_counter()
    while True:
        if throttle and source_fps > 0:
            delay = index / source_fps - (perf_counter() - started)
            if delay > 0:
                sleep(delay)
        ok, frame = capture.read()
        if not ok or frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            queue.stop()
            return
        timestamp = index / source_fps if source_fps > 0 else 0.0
        if not queue.put((index, timestamp, frame, perf_counter())):
            return
        index += 1


def run_realtime_tracking_video(
    detector: OpenCVDetector,
    source: Path | str | int,
    output: Path | str,
    *,
    tracker_type: str = "bytetrack",
    tracker_config: Path | str | None = None,
    frame_stride: int = 2,
    queue_maxsize: int = 16,
    confirm_hits: int = 3,
    vote_window: int = 5,
    lost_ttl_seconds: float = 1.5,
    predict_gap_limit_frames: int = 4,
    id_switch_iou: float = 0.5,
    max_frames: int | None = None,
    fallback_fps: float = DEFAULT_FALLBACK_FPS,
    progress_interval: int = 30,
    write_jsonl: bool = False,
    force: bool = False,
    show: bool = False,
    throttle: bool = True,
    capture_factory: Callable[..., object] = cv2.VideoCapture,
    writer_factory: Callable[..., object] = None,
) -> dict[str, object]:
    """Run the real-time M6 pipeline on a local video or camera index.

    ``throttle=True`` (default) paces the capture thread to ``source_fps`` so
    file processing emulates a live stream; pass ``throttle=False`` to decode
    as fast as possible (frames beyond the bounded queue are dropped and
    counted).
    """
    if writer_factory is None:
        def writer_factory(*args, **kwargs):  # type: ignore[misc]
            return cv2.VideoWriter(*args, **kwargs)

    source_text = str(source)
    source_candidate = Path(source_text).expanduser()
    is_camera = isinstance(source, int) or (
        isinstance(source, str) and source.strip().isdigit() and not source_candidate.exists()
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
    if frame_stride < 1 or queue_maxsize < 1:
        raise ValueError("frame_stride and queue_maxsize must be positive")
    if not math.isfinite(fallback_fps) or fallback_fps <= 0:
        raise ValueError("fallback_fps must be a finite positive number")
    if not math.isfinite(lost_ttl_seconds) or lost_ttl_seconds <= 0:
        raise ValueError("lost_ttl_seconds must be a finite positive number")

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
    source_fps = fallback_fps if (not math.isfinite(raw_fps) or raw_fps <= 0) else raw_fps
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    declared_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if width < 1 or height < 1:
        capture.release()
        released["capture_released"] = True
        raise ValueError(f"video reports invalid dimensions {width}x{height}: {input_label}")

    try:
        track_buffer = calculate_track_buffer(source_fps, frame_stride, lost_ttl_seconds)
        config = load_tracker_config(
            tracker_type,
            track_buffer=track_buffer,
            config_path=tracker_config,
            device=str(getattr(detector, "device", "cpu")),
        )
        session = TrackingSession(tracker_type=tracker_type, config=config)
        session.reset()
        realtime = RealtimeTrackingSession(
            session,
            frame_stride=frame_stride,
            confirm_hits=confirm_hits,
            vote_window=vote_window,
            vote_min_majority=min(vote_window, confirm_hits),
            predict_gap_limit_frames=predict_gap_limit_frames,
            id_switch_iou=id_switch_iou,
        )
        # Warm up the CUDA runtime before the timed pipeline so the first real
        # frames do not pay one-off kernel compilation / cuDNN benchmark /
        # graph-capture cost. Without it the first ~40 detection frames can run
        # 2x slower and dominate the P95 latency.
        try:
            warmup = np.zeros((height, width, 3), dtype=np.uint8)
            for _ in range(8):
                detector.predict_bgr(warmup)
        except Exception:
            pass
    except Exception:
        capture.release()
        released["capture_released"] = True
        raise

    queue = BoundedFrameQueue(maxsize=queue_maxsize)
    capture_thread = Thread(
        target=_capture_loop,
        args=(capture, queue, source_fps, throttle),
        daemon=True,
    )
    capture_thread.start()

    total_frames = 0
    detection_frames = 0
    interpolated_frames = 0
    total_inference = 0.0
    total_tracking = 0.0
    total_drawing = 0.0
    full_frame_ms: list[float] = []
    e2e_latency_ms: list[float] = []
    active_sum = 0
    lost_sum = 0
    class_counts = {"helmet": 0, "no_helmet": 0}
    prediction_observation_count = 0
    codec = ""
    pipeline_started = perf_counter()
    last_frame_completed = pipeline_started

    try:
        # Prefer H.264 (avc1): on this machine it encodes ~8x faster than mp4v
        # (~0.6 ms vs ~5 ms per 720p frame), which directly lowers the P95
        # full-frame latency of the detection frames.
        writer, codec = _open_writer(
            output_path,
            source_fps,
            (width, height),
            writer_factory,
            codecs=("avc1", "mp4v"),
        )
        if write_jsonl:
            jsonl_file = jsonl_path.open("w", encoding="utf-8")
        while max_frames is None or total_frames < max_frames:
            item = queue.get()
            if item is None:
                break
            frame_index, timestamp_seconds, frame, enqueue_wall = item
            frame_started = perf_counter()
            observations: list[TrackObservation] = []
            detection_performed = frame_index % frame_stride == 0
            if detection_performed:
                inference = detector.predict_bgr(frame)
                total_inference += float(inference.inference_seconds)
                tracked = realtime.update_detection(
                    inference.detections,
                    frame,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                )
                total_tracking += tracked.tracking_seconds
                observations = tracked.observations
                detection_frames += 1
                for observation in observations:
                    class_counts[observation.class_name] += 1
                    if observation.is_prediction:
                        prediction_observation_count += 1
            else:
                observations = realtime.predict_interpolated(frame_index, timestamp_seconds)
                interpolated_frames += 1
                prediction_observation_count += len(observations)
                for observation in observations:
                    class_counts[observation.class_name] += 1

            elapsed_before_draw = perf_counter() - frame_started
            processing_fps = 1.0 / elapsed_before_draw if elapsed_before_draw > 0 else 0.0
            drawing_started = perf_counter()
            output_frame = draw_tracks(
                frame,
                observations,
                active_tracks=realtime.current_active_count,
                lost_tracks=realtime.current_lost_count,
                created_tracks=realtime.statistics["created_tracks"],
                processing_fps=processing_fps,
                tracker_type=tracker_type,
                # Constant lite style on both detection and interpolated frames
                # so the overlay (labels/panel) does not alternate and flicker.
                lite=True,
            )
            total_drawing += perf_counter() - drawing_started
            writer.write(output_frame)
            active_sum += realtime.current_active_count
            lost_sum += realtime.current_lost_count
            if jsonl_file is not None:
                row = {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "detection_performed": detection_performed,
                    "observations": [item.to_dict() for item in observations],
                }
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_frames += 1
            completed = perf_counter()
            full_frame_ms.append((completed - frame_started) * 1000.0)
            e2e_latency_ms.append((completed - enqueue_wall) * 1000.0)
            last_frame_completed = completed
            if total_frames % progress_interval == 0:
                LOGGER.info(
                    "realtime tracking progress: %d frames written (%d det, %d interp)",
                    total_frames,
                    detection_frames,
                    interpolated_frames,
                )
            if show:
                cv2.imshow("helmet-safety M6 realtime", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        queue.stop()
        capture_thread.join(timeout=2.0)
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

    track_stats = realtime.finalize()
    # enqueued already counts every put() attempt, including evicted frames.
    total_read = queue.enqueued
    report: dict[str, object] = {
        "schema_version": "m6-realtime-v1",
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
            "fp16": bool(getattr(detector, "fp16", False)),
        },
        "tracker_type": tracker_type,
        "tracker_config": config.to_dict(),
        "throttle": throttle,
        "draw_lite_on_detection_frames": True,
        "width": width,
        "height": height,
        "codec": codec,
        "declared_source_frames": max(0, declared_frames),
        "total_frames": total_frames,
        "detection_frames": detection_frames,
        "interpolated_frames": interpolated_frames,
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "raw_source_fps": raw_fps if math.isfinite(raw_fps) else None,
        "source_fps": source_fps,
        "detection_fps": source_fps / frame_stride,
        "output_fps": source_fps,
        "lost_ttl_seconds": lost_ttl_seconds,
        "track_buffer": track_buffer,
        "queue": {
            "maxsize": queue_maxsize,
            "enqueued_frames": queue.enqueued,
            "dequeued_frames": queue.dequeued,
            "dropped_frames": queue.dropped,
            "total_read_frames": total_read,
            "dropped_rate": queue.dropped / total_read if total_read > 0 else 0.0,
        },
        "latency_ms": {
            "full_frame": _percentiles(full_frame_ms, [50, 95, 99]),
            "max_full_frame_ms": max(full_frame_ms, default=0.0),
            "average_full_frame_ms": (
                sum(full_frame_ms) / len(full_frame_ms) if full_frame_ms else 0.0
            ),
            "queue_e2e": _percentiles(e2e_latency_ms, [50, 95, 99]),
            "max_queue_e2e_ms": max(e2e_latency_ms, default=0.0),
            "average_queue_e2e_ms": (
                sum(e2e_latency_ms) / len(e2e_latency_ms) if e2e_latency_ms else 0.0
            ),
        },
        "throughput": {
            # Steady-state throughput up to the last written frame, excluding
            # one-time teardown (writer/capture release) so it reflects the
            # sustained processing rate rather than a single cleanup stall.
            "end_to_end_throughput_fps": (
                total_frames / (last_frame_completed - pipeline_started)
                if last_frame_completed > pipeline_started
                else 0.0
            ),
            "steady_pipeline_seconds": max(0.0, last_frame_completed - pipeline_started),
            "wall_clock_pipeline_seconds": pipeline_seconds,
        },
        "average_inference_seconds": (
            total_inference / detection_frames if detection_frames else 0.0
        ),
        "average_tracking_seconds": (
            total_tracking / detection_frames if detection_frames else 0.0
        ),
        "average_drawing_seconds": total_drawing / total_frames if total_frames else 0.0,
        "average_active_tracks": active_sum / total_frames if total_frames else 0.0,
        "average_lost_tracks": lost_sum / total_frames if total_frames else 0.0,
        "class_observation_boxes": class_counts,
        "prediction_observation_boxes": prediction_observation_count,
        "tracking_quality": track_stats,
        "resource_release_status": released,
        "note": (
            "Realtime pipeline: bounded async capture queue + detection on every "
            "frame_stride-th frame + constant-velocity interpolation on in-between "
            "frames. Tracks are gated by confirmed hits and per-track temporal class "
            "voting. short_track_ratio_detection_hits_le2 uses the tracker hit count "
            "(baseline 68%); short_track_ratio_observed_frames_le2 counts interpolated "
            "frames as visible so in-between-frame continuity is reflected. "
            "id_switch_estimate is a heuristic IoU match of new IDs against "
            "recently-lost tracks, not ground-truth MOT scoring."
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
