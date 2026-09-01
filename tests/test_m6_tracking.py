from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def _core():
    return importlib.import_module("helmet_safety.tracking.core")


def _drawing():
    return importlib.import_module("helmet_safety.tracking.drawing")


def _video():
    return importlib.import_module("helmet_safety.tracking.video")


def _detection(x1: float, y1: float, x2: float, y2: float, confidence: float = 0.9, class_id: int = 0):
    from helmet_safety.inference.opencv import Detection

    return Detection(
        class_id=class_id,
        class_name={0: "helmet", 1: "no_helmet"}[class_id],
        confidence=confidence,
        xyxy=(x1, y1, x2, y2),
    )


def _new_session(*, buffer: int = 3, tracker: str = "bytetrack"):
    core = _core()
    config = core.load_tracker_config(tracker, track_buffer=buffer)
    return core.TrackingSession(tracker_type=tracker, config=config)


def _step(session, detections, index: int):
    return session.update(
        detections,
        np.zeros((96, 128, 3), dtype=np.uint8),
        frame_index=index,
        timestamp_seconds=index / 10.0,
    )


def test_track_observation_has_stable_serializable_schema() -> None:
    core = _core()
    observation = core.TrackObservation(
        track_id=7,
        class_id=1,
        class_name="no_helmet",
        confidence=0.73,
        xyxy=(1.0, 2.0, 11.0, 22.0),
        frame_index=4,
        timestamp_seconds=0.4,
        track_state="active",
        track_age=5,
        hits=3,
        time_since_update=0,
        is_prediction=False,
    )

    assert observation.to_dict() == {
        "track_id": 7,
        "class_id": 1,
        "class_name": "no_helmet",
        "confidence": 0.73,
        "xyxy": [1.0, 2.0, 11.0, 22.0],
        "frame_index": 4,
        "timestamp_seconds": 0.4,
        "track_state": "active",
        "track_age": 5,
        "hits": 3,
        "time_since_update": 0,
        "is_prediction": False,
    }


def test_bytetrack_keeps_one_id_for_continuous_motion_and_class_fluctuation() -> None:
    session = _new_session()

    first = _step(session, [_detection(10, 10, 30, 40, class_id=0)], 0)
    second = _step(session, [_detection(12, 10, 32, 40, class_id=1)], 1)

    assert [item.track_id for item in first.observations if item.track_state == "active"] == [1]
    assert [item.track_id for item in second.observations if item.track_state == "active"] == [1]
    history = session.track_summaries()[0]["class_history"]
    assert [(item["frame_index"], item["class_id"]) for item in history] == [(0, 0), (1, 1)]


def test_bytetrack_assigns_unique_ids_to_simultaneous_targets() -> None:
    result = _step(
        _new_session(),
        [_detection(5, 5, 25, 35), _detection(80, 8, 105, 40, class_id=1)],
        0,
    )

    ids = [item.track_id for item in result.observations if item.track_state == "active"]
    assert len(ids) == 2
    assert len(set(ids)) == 2


def test_low_confidence_detection_continues_track_but_cannot_create_one() -> None:
    session = _new_session()

    first = _step(session, [_detection(10, 10, 30, 40, confidence=0.18)], 0)
    assert first.observations == []
    assert session.statistics["created_tracks"] == 0

    continued = _new_session()
    _step(continued, [_detection(10, 10, 30, 40, confidence=0.9)], 0)
    result = _step(continued, [_detection(11, 10, 31, 40, confidence=0.18)], 1)
    assert [item.track_id for item in result.observations if item.track_state == "active"] == [1]
    assert result.observations[0].confidence == pytest.approx(0.18)


def test_two_targets_keep_ids_through_crossing_and_short_occlusion() -> None:
    session = _new_session(buffer=5)
    seen_a: list[int] = []
    seen_b: list[int] = []
    for frame_index in range(24):
        detections = []
        ax = 8 + 3 * frame_index
        bx = 98 - 3 * frame_index
        if frame_index not in {9, 10}:
            detections.append(_detection(ax, 10, ax + 22, 42, class_id=frame_index % 7 == 0))
        detections.append(_detection(bx, 48, bx + 22, 80, class_id=1))
        result = _step(session, detections, frame_index)
        active = [item for item in result.observations if item.track_state == "active"]
        if frame_index not in {9, 10}:
            nearest_a = min(active, key=lambda item: abs(item.xyxy[0] - ax) + abs(item.xyxy[1] - 10))
            seen_a.append(nearest_a.track_id)
        nearest_b = min(active, key=lambda item: abs(item.xyxy[0] - bx) + abs(item.xyxy[1] - 48))
        seen_b.append(nearest_b.track_id)

    assert set(seen_a) == {1}
    assert set(seen_b) == {2}
    assert session.statistics["recovered_tracks"] >= 1


def test_empty_detection_frame_returns_lost_prediction_without_crashing() -> None:
    session = _new_session(buffer=3)
    _step(session, [_detection(10, 10, 30, 40)], 0)

    result = _step(session, [], 1)

    assert len(result.observations) == 1
    assert result.observations[0].track_id == 1
    assert result.observations[0].track_state == "lost"
    assert result.observations[0].is_prediction is True
    assert result.observations[0].time_since_update == 1


def test_track_recovers_same_id_inside_buffer() -> None:
    session = _new_session(buffer=2)
    _step(session, [_detection(10, 10, 30, 40)], 0)
    _step(session, [], 1)

    recovered = _step(session, [_detection(11, 10, 31, 40)], 2)

    assert [item.track_id for item in recovered.observations if item.track_state == "active"] == [1]
    assert session.statistics["recovered_tracks"] == 1


def test_track_gets_new_id_after_buffer_expires() -> None:
    session = _new_session(buffer=1)
    _step(session, [_detection(10, 10, 30, 40)], 0)
    _step(session, [], 1)
    _step(session, [], 2)
    _step(session, [_detection(10, 10, 30, 40)], 3)

    reappeared = _step(session, [_detection(11, 10, 31, 40)], 4)

    assert [item.track_id for item in reappeared.observations if item.track_state == "active"] == [2]
    assert session.statistics["removed_tracks"] >= 1
    assert session.statistics["created_tracks"] == 2


def test_reset_clears_state_and_restarts_id_space() -> None:
    session = _new_session()
    _step(session, [_detection(10, 10, 30, 40)], 0)

    session.reset()
    after_reset = _step(session, [_detection(70, 10, 90, 40)], 0)

    assert [item.track_id for item in after_reset.observations if item.track_state == "active"] == [1]
    assert session.statistics["created_tracks"] == 1


@pytest.mark.parametrize(
    ("fps", "stride", "ttl", "expected"),
    [(30.0, 1, 1.0, 30), (25.0, 2, 1.0, 13), (29.97, 3, 1.5, 15), (1.0, 30, 0.1, 1)],
)
def test_track_buffer_uses_detection_timebase(fps: float, stride: int, ttl: float, expected: int) -> None:
    assert _core().calculate_track_buffer(fps, stride, ttl) == expected


def test_default_tracker_configs_match_m6_baseline() -> None:
    core = _core()
    byte = core.load_tracker_config("bytetrack", track_buffer=17)
    bot = core.load_tracker_config("botsort", track_buffer=17)

    for config in (byte, bot):
        assert config.track_high_thresh == 0.25
        assert config.track_low_thresh == 0.10
        assert config.new_track_thresh == 0.25
        assert config.match_thresh == 0.80
        assert config.fuse_score is True
        assert config.track_buffer == 17
    assert byte.tracker_type == "bytetrack"
    assert bot.tracker_type == "botsort"
    assert bot.gmc_method == "sparseOptFlow"
    assert bot.with_reid is False


def test_custom_tracker_config_is_loaded_and_type_must_match(tmp_path: Path) -> None:
    config_path = tmp_path / "tracker.yaml"
    config_path.write_text(
        "tracker_type: bytetrack\ntrack_high_thresh: 0.31\nmatch_thresh: 0.72\n",
        encoding="utf-8",
    )

    config = _core().load_tracker_config("bytetrack", track_buffer=9, config_path=config_path)

    assert config.track_high_thresh == 0.31
    assert config.match_thresh == 0.72
    assert config.track_buffer == 9
    with pytest.raises(ValueError, match="tracker_type"):
        _core().load_tracker_config("botsort", track_buffer=9, config_path=config_path)


def test_draw_tracks_preserves_shape_dtype_and_marks_predictions() -> None:
    core = _core()
    drawing = _drawing()
    image = np.zeros((100, 140, 3), dtype=np.uint8)
    observations = [
        core.TrackObservation(1, 0, "helmet", 0.9, (-5, -3, 40, 50), 0, 0.0, "active", 1, 1, 0, False),
        core.TrackObservation(2, 1, "no_helmet", 0.6, (80, 50, 200, 120), 0, 0.0, "lost", 2, 1, 1, True),
    ]

    output = drawing.draw_tracks(
        image,
        observations,
        active_tracks=1,
        lost_tracks=1,
        created_tracks=2,
        processing_fps=12.5,
        tracker_type="bytetrack",
    )

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert np.any(output != image)
    assert not np.any(image)


class _FakeDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.weights = Path("fake.pt")
        self.device = "cpu"
        self.imgsz = 960
        self.conf = 0.10
        self.iou = 0.70
        self.max_det = 300

    def predict_bgr(self, image: np.ndarray):
        from helmet_safety.inference.opencv import InferenceResult

        self.calls += 1
        x = 10 + self.calls
        return InferenceResult([_detection(x, 10, x + 20, 40)], 0.002)


def _write_video(path: Path, frames: int = 5, fps: float = 8.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


def test_video_pipeline_counts_stride_writes_jsonl_and_initializes_tracker_once(tmp_path: Path) -> None:
    video = _video()
    core = _core()
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _write_video(source, frames=5, fps=8.0)
    detector = _FakeDetector()
    created = []

    class FakeTracker:
        def __init__(self) -> None:
            self.statistics = {
                "created_tracks": 0,
                "lost_tracks": 0,
                "recovered_tracks": 0,
                "removed_tracks": 0,
            }
            self.current_active_count = 0
            self.current_lost_count = 0
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

        def update(self, detections, image, *, frame_index, timestamp_seconds):
            return core.FrameTrackingResult([], 0.001, 0, 0, 0)

        def track_summaries(self):
            return []

    def tracker_factory(**kwargs):
        tracker = FakeTracker()
        created.append((kwargs, tracker))
        return tracker

    report = video.run_tracking_video(
        detector,
        source,
        output,
        frame_stride=2,
        lost_ttl_seconds=1.0,
        write_jsonl=True,
        tracker_factory=tracker_factory,
    )

    assert detector.calls == 3
    assert len(created) == 1
    assert created[0][0]["config"].track_buffer == 4
    assert created[0][1].reset_calls == 1
    assert report["total_frames"] == 5
    assert report["detection_frames"] == 3
    assert report["skipped_frames"] == 2
    assert report["source_fps"] == pytest.approx(8.0, rel=0.05)
    assert report["detection_fps"] == pytest.approx(4.0, rel=0.05)
    assert report["track_buffer"] == 4
    assert report["resource_release_status"] == {
        "capture_released": True,
        "writer_released": True,
        "jsonl_closed": True,
        "windows_destroyed": True,
    }
    rows = [json.loads(line) for line in output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert [row["detection_performed"] for row in rows] == [True, False, True, False, True]
    assert rows[1]["observations"] == []
    assert output.with_suffix(".json").is_file()


def test_video_pipeline_rejects_missing_input_and_all_existing_outputs(tmp_path: Path) -> None:
    video = _video()
    with pytest.raises(FileNotFoundError, match="input"):
        video.run_tracking_video(_FakeDetector(), tmp_path / "missing.mp4", tmp_path / "out.mp4")

    source = tmp_path / "input.mp4"
    _write_video(source, frames=1)
    for suffix in (".mp4", ".json", ".jsonl"):
        target = tmp_path / f"output{suffix}"
        target.write_bytes(b"existing")
        with pytest.raises(FileExistsError, match="--force"):
            video.run_tracking_video(
                _FakeDetector(), source, tmp_path / "output.mp4", write_jsonl=suffix == ".jsonl"
            )
        target.unlink()


def test_video_capture_and_writer_are_released_on_success_and_failure(tmp_path: Path) -> None:
    video = _video()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"placeholder")

    class Capture:
        def __init__(self) -> None:
            self.released = False
            self.reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(prop, 0)

        def read(self):
            self.reads += 1
            return (True, np.zeros((48, 64, 3), dtype=np.uint8)) if self.reads == 1 else (False, None)

        def release(self):
            self.released = True

    class Writer:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            self.released = True

    class ExplodingDetector(_FakeDetector):
        def predict_bgr(self, image):
            raise RuntimeError("boom")

    capture = Capture()
    writer = Writer()
    with pytest.raises(RuntimeError, match="boom"):
        video.run_tracking_video(
            ExplodingDetector(),
            source,
            tmp_path / "failed.mp4",
            capture_factory=lambda _: capture,
            writer_factory=lambda *_: writer,
        )
    assert capture.released
    assert writer.released

    capture2 = Capture()
    writer2 = Writer()
    report = video.run_tracking_video(
        _FakeDetector(),
        source,
        tmp_path / "success.mp4",
        capture_factory=lambda _: capture2,
        writer_factory=lambda *_: writer2,
    )
    assert report["resource_release_status"]["capture_released"] is True
    assert capture2.released
    assert writer2.released


def test_capture_is_released_when_tracker_initialization_fails(tmp_path: Path) -> None:
    video = _video()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"placeholder")

    class Capture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(prop, 0)

        def release(self):
            self.released = True

    capture = Capture()

    def fail_tracker(**kwargs):
        raise RuntimeError("tracker init failed")

    with pytest.raises(RuntimeError, match="tracker init failed"):
        video.run_tracking_video(
            _FakeDetector(),
            source,
            tmp_path / "output.mp4",
            capture_factory=lambda _: capture,
            tracker_factory=fail_tracker,
        )

    assert capture.released


def test_numeric_camera_source_uses_an_isolated_tracker_lifecycle(tmp_path: Path) -> None:
    video = _video()
    opened_sources = []
    tracker_constructions = []

    class Capture:
        def __init__(self) -> None:
            self.reads = 0

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 0,
            }.get(prop, 0)

        def read(self):
            self.reads += 1
            return (True, np.zeros((48, 64, 3), dtype=np.uint8)) if self.reads == 1 else (False, None)

        def release(self):
            pass

    class Writer:
        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    def capture_factory(source):
        opened_sources.append(source)
        return Capture()

    def tracker_factory(**kwargs):
        tracker_constructions.append(kwargs)
        return _new_session(buffer=kwargs["config"].track_buffer)

    report = video.run_tracking_video(
        _FakeDetector(),
        0,
        tmp_path / "camera.mp4",
        max_frames=1,
        capture_factory=capture_factory,
        writer_factory=lambda *_: Writer(),
        tracker_factory=tracker_factory,
    )

    assert opened_sources == [0]
    assert len(tracker_constructions) == 1
    assert report["input_path"] == "camera:0"
