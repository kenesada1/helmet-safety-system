"""Tracker state and Ultralytics ByteTrack/BoT-SORT adaptation for M6."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import yaml

from helmet_safety.inference.opencv import CLASS_NAMES, Detection


SUPPORTED_TRACKERS = frozenset({"bytetrack", "botsort"})
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATHS = {
    "bytetrack": PROJECT_ROOT / "configs" / "m6_bytetrack.yaml",
    "botsort": PROJECT_ROOT / "configs" / "m6_botsort.yaml",
}


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    tracker_type: str
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10
    new_track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.80
    fuse_score: bool = True
    gmc_method: str = "sparseOptFlow"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.8
    with_reid: bool = False
    model: str = "auto"
    device: str = "cpu"

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**asdict(self))

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        if self.tracker_type == "bytetrack":
            for key in (
                "gmc_method",
                "proximity_thresh",
                "appearance_thresh",
                "with_reid",
                "model",
                "device",
            ):
                result.pop(key)
        return result


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    frame_index: int
    timestamp_seconds: float
    track_state: str
    track_age: int
    hits: int
    time_since_update: int
    is_prediction: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "track_state": self.track_state,
            "track_age": self.track_age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
            "is_prediction": self.is_prediction,
        }


@dataclass(frozen=True, slots=True)
class FrameTrackingResult:
    observations: list[TrackObservation]
    tracking_seconds: float
    active_tracks: int
    lost_tracks: int
    created_tracks: int


@dataclass(slots=True)
class _TrackMetadata:
    hits: int
    first_tracker_frame: int
    first_source_frame: int
    class_history: list[dict[str, object]] = field(default_factory=list)


class _TrackerDetections:
    """The small part of Ultralytics Boxes consumed by its tracker classes."""

    def __init__(self, detections: Sequence[Detection], image_shape: tuple[int, int]) -> None:
        height, width = image_shape
        boxes: list[tuple[float, float, float, float]] = []
        confidences: list[float] = []
        classes: list[float] = []
        for detection in detections:
            coords = np.asarray(detection.xyxy, dtype=np.float32)
            if coords.shape != (4,) or not np.all(np.isfinite(coords)):
                continue
            x1, x2 = sorted((float(coords[0]), float(coords[2])))
            y1, y2 = sorted((float(coords[1]), float(coords[3])))
            x1 = min(max(x1, 0.0), float(width - 1))
            x2 = min(max(x2, 0.0), float(width - 1))
            y1 = min(max(y1, 0.0), float(height - 1))
            y2 = min(max(y2, 0.0), float(height - 1))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            boxes.append((x1, y1, x2, y2))
            confidences.append(float(detection.confidence))
            classes.append(float(detection.class_id))
        self.xyxy = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confidences, dtype=np.float32)
        self.cls = np.asarray(classes, dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        converted = self.xyxy.copy()
        converted[:, 2:] -= converted[:, :2]
        converted[:, :2] += converted[:, 2:] / 2.0
        return converted

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index: object) -> "_TrackerDetections":
        subset = object.__new__(type(self))
        subset.xyxy = np.asarray(self.xyxy[index], dtype=np.float32).reshape(-1, 4)
        subset.conf = np.asarray(self.conf[index], dtype=np.float32).reshape(-1)
        subset.cls = np.asarray(self.cls[index], dtype=np.float32).reshape(-1)
        return subset


def calculate_track_buffer(source_fps: float, frame_stride: int, lost_ttl_seconds: float) -> int:
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be a finite positive number")
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    if not math.isfinite(lost_ttl_seconds) or lost_ttl_seconds <= 0:
        raise ValueError("lost_ttl_seconds must be a finite positive number")
    return max(1, math.ceil(lost_ttl_seconds * source_fps / frame_stride))


def _validate_threshold(value: object, name: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")
    return number


def load_tracker_config(
    tracker_type: str,
    *,
    track_buffer: int,
    config_path: Path | str | None = None,
    device: str = "cpu",
) -> TrackerConfig:
    normalized = tracker_type.lower()
    if normalized not in SUPPORTED_TRACKERS:
        raise ValueError(f"tracker must be one of {sorted(SUPPORTED_TRACKERS)}")
    if track_buffer < 1:
        raise ValueError("track_buffer must be positive")
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATHS[normalized]
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"tracker config does not exist: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"tracker config must contain a YAML mapping: {path}")
    configured_type = str(raw.get("tracker_type", normalized)).lower()
    if configured_type != normalized:
        raise ValueError(
            f"tracker_type mismatch: requested {normalized!r}, config contains {configured_type!r}"
        )
    defaults: dict[str, object] = asdict(TrackerConfig(tracker_type=normalized))
    unknown = set(raw) - set(defaults)
    if unknown:
        raise ValueError(f"unsupported tracker config keys: {sorted(unknown)}")
    defaults.update(raw)
    defaults["tracker_type"] = normalized
    defaults["track_buffer"] = int(track_buffer)
    defaults["device"] = str(device)
    for key in (
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "match_thresh",
        "proximity_thresh",
        "appearance_thresh",
    ):
        defaults[key] = _validate_threshold(defaults[key], key)
    if defaults["track_low_thresh"] > defaults["track_high_thresh"]:
        raise ValueError("track_low_thresh cannot exceed track_high_thresh")
    return TrackerConfig(**defaults)


class TrackingSession:
    """One isolated tracker lifecycle; create or reset it for every input stream."""

    def __init__(self, *, tracker_type: str, config: TrackerConfig) -> None:
        if config.tracker_type != tracker_type:
            raise ValueError("tracker_type and config.tracker_type must match")
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.trackers.byte_tracker import BYTETracker

        tracker_class = BYTETracker if tracker_type == "bytetrack" else BOTSORT
        args = config.to_namespace()
        self.tracker_type = tracker_type
        self.config = config
        # Ultralytics 8.4.120 consumes track_buffer directly in tracker-update
        # units. M6 computes that value from source FPS/stride before this point.
        self._tracker = tracker_class(args)
        self._metadata: dict[int, _TrackMetadata] = {}
        self._seen_ids: set[int] = set()
        self._previous_active: set[int] = set()
        self._previous_lost: set[int] = set()
        self._previous_removed: set[int] = set()
        self.statistics: dict[str, int] = {}
        self.reset()

    @property
    def current_active_count(self) -> int:
        return len(self._previous_active)

    @property
    def current_lost_count(self) -> int:
        return len(self._previous_lost)

    def reset(self) -> None:
        self._tracker.reset()
        self._metadata.clear()
        self._seen_ids.clear()
        self._previous_active.clear()
        self._previous_lost.clear()
        self._previous_removed.clear()
        self.statistics = {
            "created_tracks": 0,
            "lost_tracks": 0,
            "recovered_tracks": 0,
            "removed_tracks": 0,
        }

    @staticmethod
    def _track_id(track: object) -> int:
        return int(getattr(track, "track_id"))

    def _record_detection(self, track: object, frame_index: int, timestamp_seconds: float, *, is_new: bool) -> None:
        track_id = self._track_id(track)
        if is_new:
            metadata = _TrackMetadata(
                hits=1,
                first_tracker_frame=int(getattr(track, "start_frame")),
                first_source_frame=frame_index,
            )
            self._metadata[track_id] = metadata
        else:
            metadata = self._metadata[track_id]
            metadata.hits += 1
        class_id = int(getattr(track, "cls"))
        metadata.class_history.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, str(class_id)),
                "confidence": float(getattr(track, "score")),
            }
        )

    def update(
        self,
        detections: Sequence[Detection],
        image: np.ndarray,
        *,
        frame_index: int,
        timestamp_seconds: float,
    ) -> FrameTrackingResult:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("tracker image must be a non-empty BGR array")
        if frame_index < 0 or timestamp_seconds < 0:
            raise ValueError("frame index and timestamp must be non-negative")
        tracker_input = _TrackerDetections(detections, image.shape[:2])
        started = perf_counter()
        self._tracker.update(tracker_input, image)
        tracking_seconds = perf_counter() - started

        # Ultralytics 8.4.120 appends the current frame's removals after it
        # filters the lost pool, leaving an expired item associable for one
        # extra update. Prune that transient overlap so track_buffer means the
        # documented number of detection updates, without an off-by-one leak.
        just_removed = {
            self._track_id(track)
            for track in getattr(self._tracker, "removed_stracks_frame", [])
        }
        if just_removed:
            self._tracker.lost_stracks = [
                track
                for track in self._tracker.lost_stracks
                if self._track_id(track) not in just_removed
            ]

        active_tracks = [
            track
            for track in self._tracker.tracked_stracks
            if bool(getattr(track, "is_activated", False))
        ]
        lost_tracks = list(self._tracker.lost_stracks)
        removed_tracks = list(self._tracker.removed_stracks)
        active_ids = {self._track_id(track) for track in active_tracks}
        lost_ids = {self._track_id(track) for track in lost_tracks}
        removed_ids = {self._track_id(track) for track in removed_tracks}

        all_tracks: dict[int, object] = {}
        for track in [*self._tracker.tracked_stracks, *lost_tracks, *removed_tracks]:
            all_tracks[self._track_id(track)] = track
        new_ids = set(all_tracks) - self._seen_ids
        for track_id in sorted(new_ids):
            self._record_detection(
                all_tracks[track_id], frame_index, timestamp_seconds, is_new=True
            )
        for track in active_tracks:
            track_id = self._track_id(track)
            if track_id not in new_ids and int(getattr(track, "frame_id")) == int(self._tracker.frame_id):
                self._record_detection(track, frame_index, timestamp_seconds, is_new=False)

        self.statistics["created_tracks"] += len(new_ids)
        self.statistics["lost_tracks"] += len(self._previous_active & lost_ids)
        self.statistics["recovered_tracks"] += len(self._previous_lost & active_ids)
        self.statistics["removed_tracks"] += len(removed_ids - self._previous_removed)
        self._seen_ids.update(new_ids)
        self._previous_active = active_ids
        self._previous_lost = lost_ids
        self._previous_removed = removed_ids

        observations: list[TrackObservation] = []
        for state, tracks in (("active", active_tracks), ("lost", lost_tracks)):
            for track in sorted(tracks, key=self._track_id):
                track_id = self._track_id(track)
                metadata = self._metadata[track_id]
                class_id = int(getattr(track, "cls"))
                xyxy = tuple(float(value) for value in np.asarray(getattr(track, "xyxy")).reshape(4))
                observations.append(
                    TrackObservation(
                        track_id=track_id,
                        class_id=class_id,
                        class_name=CLASS_NAMES.get(class_id, str(class_id)),
                        confidence=float(getattr(track, "score")),
                        xyxy=xyxy,  # type: ignore[arg-type]
                        frame_index=frame_index,
                        timestamp_seconds=float(timestamp_seconds),
                        track_state=state,
                        track_age=int(self._tracker.frame_id) - metadata.first_tracker_frame + 1,
                        hits=metadata.hits,
                        time_since_update=int(self._tracker.frame_id) - int(getattr(track, "frame_id")),
                        is_prediction=state != "active",
                    )
                )
        return FrameTrackingResult(
            observations=observations,
            tracking_seconds=tracking_seconds,
            active_tracks=len(active_ids),
            lost_tracks=len(lost_ids),
            created_tracks=len(self._seen_ids),
        )

    def track_summaries(self) -> list[dict[str, object]]:
        return [
            {
                "track_id": track_id,
                "hits": metadata.hits,
                "first_source_frame": metadata.first_source_frame,
                "class_history": list(metadata.class_history),
            }
            for track_id, metadata in sorted(self._metadata.items())
        ]
