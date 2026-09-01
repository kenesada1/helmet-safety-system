from __future__ import annotations

import importlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


def _realtime():
    return importlib.import_module("helmet_safety.tracking.realtime")


def _core():
    return importlib.import_module("helmet_safety.tracking.core")


def _opencv():
    return importlib.import_module("helmet_safety.inference.opencv")


def _detection(x1, y1, x2, y2, confidence=0.9, class_id=0):
    from helmet_safety.inference.opencv import Detection

    return Detection(
        class_id=class_id,
        class_name={0: "helmet", 1: "no_helmet"}[class_id],
        confidence=confidence,
        xyxy=(x1, y1, x2, y2),
    )


def _realtime_session(buffer=4, **kwargs):
    core = _core()
    config = core.load_tracker_config("bytetrack", track_buffer=buffer)
    session = core.TrackingSession(tracker_type="bytetrack", config=config)
    rt = _realtime().RealtimeTrackingSession(
        session,
        frame_stride=kwargs.pop("frame_stride", 2),
        confirm_hits=kwargs.pop("confirm_hits", 3),
        vote_window=kwargs.pop("vote_window", 5),
        vote_min_majority=kwargs.pop("vote_min_majority", 3),
        predict_gap_limit_frames=kwargs.pop("predict_gap_limit_frames", 4),
        id_switch_iou=kwargs.pop("id_switch_iou", 0.5),
        **kwargs,
    )
    return rt


def _image():
    return np.zeros((96, 128, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# BoundedFrameQueue
# ---------------------------------------------------------------------------


def test_bounded_queue_evicts_oldest_and_counts_drops():
    realtime = _realtime()
    queue = realtime.BoundedFrameQueue(maxsize=2)
    frames = [np.full((4, 4, 3), index, dtype=np.uint8) for index in range(5)]
    for index, frame in enumerate(frames):
        assert queue.put((index, index / 10.0, frame, index * 0.001)) is True
    assert queue.enqueued == 5
    assert queue.dropped == 3

    # Only the freshest 2 frames survive; first dequeued is the oldest kept.
    assert queue.get()[0] == 3
    assert queue.get()[0] == 4
    queue.stop()
    assert queue.get() is None


def test_bounded_queue_stop_unblocks_get():
    realtime = _realtime()
    queue = realtime.BoundedFrameQueue(maxsize=1)
    queue.stop()
    assert queue.get() is None
    # put after stop returns False (capture loop should exit).
    assert queue.put((0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8), 0.0)) is False


def test_bounded_queue_rejects_invalid_maxsize():
    realtime = _realtime()
    with pytest.raises(ValueError, match="maxsize"):
        realtime.BoundedFrameQueue(maxsize=0)


# ---------------------------------------------------------------------------
# TemporalClassVoter
# ---------------------------------------------------------------------------


def test_voter_needs_majority_before_setting_stable_class():
    realtime = _realtime()
    voter = realtime.TemporalClassVoter(window=5, min_majority=3)
    assert voter.observe(0) is False
    assert voter.observe(0) is False
    assert voter.observe(1) is False
    assert voter.stable_class is None
    assert voter.observe(0) is True  # 0 appears 3 times -> stable 0, a switch
    assert voter.stable_class == 0
    assert voter.switches == 1


def test_voter_ignores_single_frame_flip_until_majority():
    realtime = _realtime()
    voter = realtime.TemporalClassVoter(window=5, min_majority=3)
    for _ in range(3):
        voter.observe(0)
    assert voter.stable_class == 0

    # One flipped observation is not enough to flip the stable class.
    assert voter.observe(1) is False
    assert voter.stable_class == 0
    assert voter.observe(1) is False
    assert voter.stable_class == 0
    # Third consecutive 1 in the window flips it.
    assert voter.observe(1) is True
    assert voter.stable_class == 1
    assert voter.switches == 2


# ---------------------------------------------------------------------------
# TrackMotionBuffer
# ---------------------------------------------------------------------------


def test_motion_buffer_constant_velocity_extrapolation():
    realtime = _realtime()
    buffer = realtime.TrackMotionBuffer()
    assert buffer.predict(0) is None
    buffer.update(0, (10.0, 10.0, 20.0, 30.0))
    # Single point: held still.
    assert buffer.predict(2) == pytest.approx((10.0, 10.0, 20.0, 30.0))
    buffer.update(2, (12.0, 12.0, 22.0, 32.0))
    # Velocity is +1 per frame on all corners -> at frame 4 -> +2.
    assert buffer.predict(4) == pytest.approx((14.0, 14.0, 24.0, 34.0))


# ---------------------------------------------------------------------------
# RealtimeTrackingSession
# ---------------------------------------------------------------------------


def test_interpolated_frame_emits_prediction_without_polluting_tracker():
    rt = _realtime_session(frame_stride=2)
    rt.update_detection([_detection(10, 10, 30, 40)], _image(), frame_index=0, timestamp_seconds=0.0)
    assert len(rt.tracks) == 1
    track_id = next(iter(rt.tracks))

    predicted = rt.predict_interpolated(1, 1 / 25.0)
    assert len(predicted) == 1
    assert predicted[0].track_id == track_id
    assert predicted[0].is_prediction is True
    assert predicted[0].xyxy == pytest.approx((10.0, 10.0, 30.0, 40.0))

    # Tracker state untouched by the interpolated frame: hits stayed at 1.
    assert rt.tracks[track_id].hits == 1
    assert rt.tracks[track_id].observed_frames == 2


def test_confirmed_tracks_require_confirm_hits():
    rt = _realtime_session(confirm_hits=3, frame_stride=1)
    for index in range(2):
        rt.update_detection([_detection(10, 10, 30, 40)], _image(), frame_index=index, timestamp_seconds=index / 10.0)
    assert rt.tracks[next(iter(rt.tracks))].hits == 2
    quality = rt.finalize()
    assert quality["confirmed_tracks"] == 0
    assert quality["tentative_tracks"] == 1

    rt.update_detection([_detection(11, 10, 31, 40)], _image(), frame_index=2, timestamp_seconds=0.2)
    quality = rt.finalize()
    assert quality["confirmed_tracks"] == 1


def test_class_voting_reduces_class_switches_and_short_tracks():
    rt = _realtime_session(frame_stride=1, confirm_hits=3, vote_window=5, vote_min_majority=3)
    for index in range(8):
        class_id = 0 if index != 3 else 1  # single-frame flip at frame 3
        rt.update_detection(
            [_detection(10, 10, 30, 40, class_id=class_id)],
            _image(),
            frame_index=index,
            timestamp_seconds=index / 10.0,
        )
    track = next(iter(rt.tracks.values()))
    assert track.stable_class_id == 0  # the flip did not stick
    quality = rt.finalize()
    assert quality["confirmed_tracks"] == 1
    assert quality["short_tracks_detection_hits_le2"] == 0


def test_id_switch_estimate_counts_new_id_overlapping_recent_lost():
    rt = _realtime_session(buffer=1, frame_stride=1, predict_gap_limit_frames=4)
    rt.update_detection([_detection(10, 10, 40, 40)], _image(), frame_index=0, timestamp_seconds=0.0)
    rt.update_detection([], _image(), frame_index=1, timestamp_seconds=0.1)
    rt.update_detection([], _image(), frame_index=2, timestamp_seconds=0.2)  # old id removed
    assert rt.id_switch_estimate == 0
    # New id appears but is still unactivated in ByteTrack's first frame.
    rt.update_detection([_detection(12, 12, 42, 42)], _image(), frame_index=3, timestamp_seconds=0.3)
    assert rt.id_switch_estimate == 0
    # Second detection confirms the new id; it overlaps the recently-lost track.
    rt.update_detection([_detection(13, 13, 43, 43)], _image(), frame_index=4, timestamp_seconds=0.4)
    assert rt.id_switch_estimate == 1


def test_recovered_and_removed_statistics_propagate():
    rt = _realtime_session(buffer=3, frame_stride=1)
    rt.update_detection([_detection(10, 10, 40, 40)], _image(), frame_index=0, timestamp_seconds=0.0)
    rt.update_detection([], _image(), frame_index=1, timestamp_seconds=0.1)
    rt.update_detection([_detection(12, 12, 42, 42)], _image(), frame_index=2, timestamp_seconds=0.2)
    quality = rt.finalize()
    assert quality["recovered_tracks"] >= 1
    assert quality["created_tracks"] == 1


# ---------------------------------------------------------------------------
# run_realtime_tracking_video end-to-end
# ---------------------------------------------------------------------------


class _FakeDetector:
    def __init__(self, move: bool = True) -> None:
        self.calls = 0
        self.weights = Path("fake.pt")
        self.device = "0"
        self.imgsz = 960
        self.conf = 0.10
        self.iou = 0.70
        self.max_det = 300
        self.move = move

    def predict_bgr(self, image: np.ndarray):
        from helmet_safety.inference.opencv import InferenceResult

        self.calls += 1
        x = 10 + self.calls * 3 if self.move else 10
        return InferenceResult([_detection(x, 10, x + 20, 40)], 0.002)


def _write_video(path: Path, frames: int = 7, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(frames):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()


def test_realtime_video_pipeline_stride_interpolation_and_report(tmp_path: Path) -> None:
    realtime = _realtime()
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _write_video(source, frames=7, fps=10.0)
    detector = _FakeDetector()

    report = realtime.run_realtime_tracking_video(
        detector,
        source,
        output,
        frame_stride=2,
        queue_maxsize=16,
        write_jsonl=True,
        lost_ttl_seconds=1.0,
    )

    assert report["total_frames"] == 7
    # frames 0,2,4,6 are detection frames; 1,3,5 are interpolated.
    assert report["detection_frames"] == 4
    assert report["interpolated_frames"] == 3
    assert report["tracking_quality"]["created_tracks"] == 1
    assert report["queue"]["dropped_frames"] >= 0
    assert report["latency_ms"]["full_frame"]["p95"] > 0
    assert report["latency_ms"]["queue_e2e"]["p95"] > 0
    assert report["throughput"]["end_to_end_throughput_fps"] > 0
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    rows = [
        json.loads(line)
        for line in output.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 7
    assert [row["detection_performed"] for row in rows] == [True, False, True, False, True, False, True]
    # Interpolated rows carry a prediction observation.
    assert rows[1]["observations"][0]["is_prediction"] is True


def test_realtime_video_pipeline_rejects_existing_outputs(tmp_path: Path) -> None:
    realtime = _realtime()
    source = tmp_path / "input.mp4"
    _write_video(source, frames=3)
    output = tmp_path / "output.mp4"
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="--force"):
        realtime.run_realtime_tracking_video(_FakeDetector(), source, output)


def test_realtime_video_pipeline_releases_resources_on_exception(tmp_path: Path) -> None:
    realtime = _realtime()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"placeholder")

    class Capture:
        released = False
        reads = 0

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
        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    class ExplodingDetector(_FakeDetector):
        def predict_bgr(self, image):
            raise RuntimeError("boom")

    capture = Capture()
    with pytest.raises(RuntimeError, match="boom"):
        realtime.run_realtime_tracking_video(
            ExplodingDetector(),
            source,
            tmp_path / "failed.mp4",
            capture_factory=lambda _: capture,
            writer_factory=lambda *_, **__: Writer(),
        )
    assert capture.released
