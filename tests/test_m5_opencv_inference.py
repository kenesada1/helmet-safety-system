from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


def _core() -> object:
    try:
        from helmet_safety.inference import opencv as module
    except ModuleNotFoundError:
        pytest.fail("helmet_safety.inference.opencv must implement the M5 core")
    return module


def _images() -> object:
    try:
        from helmet_safety.inference import images as module
    except ModuleNotFoundError:
        pytest.fail("helmet_safety.inference.images must implement the M5 image flow")
    return module


def _videos() -> object:
    try:
        from helmet_safety.inference import videos as module
    except ModuleNotFoundError:
        pytest.fail("helmet_safety.inference.videos must implement the M5 video flow")
    return module


class FakeDetector:
    def __init__(self) -> None:
        core = _core()
        self.calls = 0
        self._result_type = core.InferenceResult
        self._detection_type = core.Detection

    def predict_bgr(self, image: np.ndarray) -> object:
        self.calls += 1
        return self._result_type(
            detections=[
                self._detection_type(0, "helmet", 0.9, (8.0, 8.0, 30.0, 30.0)),
                self._detection_type(1, "no_helmet", 0.8, (34.0, 8.0, 55.0, 30.0)),
            ],
            inference_seconds=0.01,
        )


def test_ultralytics_conversion_returns_project_detection_records() -> None:
    core = _core()
    result = SimpleNamespace(
        boxes=SimpleNamespace(
            cls=np.array([0.0, 1.0]),
            conf=np.array([0.91, 0.82]),
            xyxy=np.array([[1.0, 2.0, 30.0, 40.0], [5.5, 6.5, 20.5, 25.5]]),
        ),
        names={0: "helmet", 1: "no_helmet"},
    )

    detections = core.detections_from_ultralytics(result)

    assert [item.to_dict() for item in detections] == [
        {
            "class_id": 0,
            "class_name": "helmet",
            "confidence": pytest.approx(0.91),
            "xyxy": [1.0, 2.0, 30.0, 40.0],
        },
        {
            "class_id": 1,
            "class_name": "no_helmet",
            "confidence": pytest.approx(0.82),
            "xyxy": [5.5, 6.5, 20.5, 25.5],
        },
    ]


def test_detector_passes_m5_parameters_to_one_reused_model(tmp_path: Path) -> None:
    core = _core()
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")

    class FakeModel:
        names = {0: "helmet", 1: "no_helmet"}

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def predict(self, **kwargs: object) -> list[object]:
            self.calls.append(kwargs)
            return [
                SimpleNamespace(
                    boxes=SimpleNamespace(
                        cls=np.array([], dtype=float),
                        conf=np.array([], dtype=float),
                        xyxy=np.empty((0, 4), dtype=float),
                    ),
                    names=self.names,
                )
            ]

    fake_model = FakeModel()
    factory_calls: list[Path] = []

    def factory(path: str) -> FakeModel:
        factory_calls.append(Path(path))
        return fake_model

    detector = core.OpenCVDetector(
        weights=weights,
        device="cpu",
        imgsz=960,
        conf=0.25,
        iou=0.70,
        max_det=300,
        model_factory=factory,
    )
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    detector.predict_bgr(image)
    detector.predict_bgr(image)

    assert factory_calls == [weights.resolve()]
    assert len(fake_model.calls) == 2
    assert fake_model.calls[0] == {
        "source": image,
        "device": "cpu",
        "imgsz": 960,
        "conf": 0.25,
        "iou": 0.70,
        "max_det": 300,
        "verbose": False,
    }


def test_default_ultralytics_loader_declares_detection_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    core = _core()
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"fake")
    constructor_calls: list[tuple[str, str | None]] = []

    class FakeModel:
        names = {0: "helmet", 1: "no_helmet"}

    def fake_yolo(path: str, *, task: str | None = None) -> FakeModel:
        constructor_calls.append((path, task))
        return FakeModel()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=fake_yolo))

    core.OpenCVDetector(weights=weights, device="cpu")

    assert constructor_calls == [(str(weights.resolve()), "detect")]


def test_detector_rejects_missing_weights_and_invalid_bgr_input(tmp_path: Path) -> None:
    core = _core()
    with pytest.raises(FileNotFoundError, match="weights"):
        core.OpenCVDetector(weights=tmp_path / "missing.pt", model_factory=lambda _: object())

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    detector = core.OpenCVDetector(weights=weights, model=object())
    with pytest.raises(ValueError, match="BGR"):
        detector.predict_bgr(np.zeros((10, 10), dtype=np.uint8))


def test_drawing_preserves_shape_dtype_and_uses_class_colors_and_counts() -> None:
    core = _core()
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    detections = [
        core.Detection(0, "helmet", 0.91, (80.0, 70.0, 130.0, 120.0)),
        core.Detection(1, "no_helmet", 0.82, (150.0, 70.0, 205.0, 125.0)),
    ]

    drawn, counts = core.draw_detections(image, detections, processing_fps=42.5)

    assert drawn.shape == image.shape
    assert drawn.dtype == image.dtype
    assert counts == {"helmet": 1, "no_helmet": 1}
    assert tuple(drawn[70, 100]) == core.CLASS_COLORS["helmet"]
    assert tuple(drawn[70, 175]) == core.CLASS_COLORS["no_helmet"]
    assert not np.shares_memory(drawn, image)


def test_drawing_empty_results_and_clips_out_of_bounds_tiny_boxes() -> None:
    core = _core()
    image = np.zeros((50, 70, 3), dtype=np.uint8)

    empty, empty_counts = core.draw_detections(image, [], processing_fps=0.0)
    clipped, clipped_counts = core.draw_detections(
        image,
        [
            core.Detection(0, "helmet", 0.75, (-20.0, -10.0, 2.0, 1.0)),
            core.Detection(1, "no_helmet", 0.65, (69.8, 49.8, 100.0, 90.0)),
        ],
        processing_fps=10.0,
    )

    assert empty.shape == image.shape
    assert empty_counts == {"helmet": 0, "no_helmet": 0}
    assert clipped.shape == image.shape
    assert clipped_counts == {"helmet": 1, "no_helmet": 1}
    assert np.any(clipped != image)


def test_image_collection_filters_supported_extensions_case_insensitively(tmp_path: Path) -> None:
    images = _images()
    for name in ("a.jpg", "b.JPEG", "c.png", "d.BMP", "ignore.gif", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "nested.jpg").write_bytes(b"x")

    paths = images.collect_image_paths(tmp_path)

    assert [path.name for path in paths] == ["a.jpg", "b.JPEG", "c.png", "d.BMP"]


def test_image_flow_rejects_missing_input_and_existing_outputs(tmp_path: Path) -> None:
    images = _images()
    detector = FakeDetector()
    with pytest.raises(FileNotFoundError, match="input"):
        images.run_image_inference(detector, tmp_path / "missing.jpg", tmp_path / "out")

    source = tmp_path / "input.jpg"
    assert cv2.imwrite(str(source), np.zeros((48, 64, 3), dtype=np.uint8))
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / source.name).write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--force"):
        images.run_image_inference(detector, source, output_dir)
    assert detector.calls == 0


def test_single_image_flow_writes_annotated_image_and_utf8_json(tmp_path: Path) -> None:
    images = _images()
    source = tmp_path / "输入.jpg"
    encoded, buffer = cv2.imencode(
        ".jpg", np.zeros((48, 64, 3), dtype=np.uint8)
    )
    assert encoded
    buffer.tofile(str(source))
    output_dir = tmp_path / "结果"

    report = images.run_image_inference(FakeDetector(), source, output_dir)

    output_image = output_dir / source.name
    report_path = output_dir / "输入.json"
    reopened = cv2.imdecode(
        np.fromfile(str(output_image), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert reopened.shape == (48, 64, 3)
    assert report_path.is_file()
    assert "输入.jpg" in report_path.read_text(encoding="utf-8")
    assert report["summary"] == {
        "successful": 1,
        "failed": 0,
        "total_detections": 2,
        "helmet": 1,
        "no_helmet": 1,
        "average_inference_seconds": pytest.approx(0.01),
    }


def _write_test_video(path: Path, *, frames: int = 5, fps: float = 8.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (64, 48),
    )
    assert writer.isOpened()
    try:
        for index in range(frames):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_flow_writes_all_frames_and_applies_frame_stride(tmp_path: Path) -> None:
    videos = _videos()
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    _write_test_video(source, frames=5, fps=8.0)
    detector = FakeDetector()

    report = videos.run_video_inference(
        detector,
        source,
        output,
        frame_stride=2,
        max_frames=5,
        progress_interval=2,
    )

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
        assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(8.0, rel=0.05)
    finally:
        capture.release()
    assert detector.calls == 3
    assert report["total_frames"] == 5
    assert report["processed_frames"] == 3
    assert report["skipped_frames"] == 2
    assert report["helmet_detections"] == 3
    assert report["no_helmet_detections"] == 3
    assert output.with_suffix(".json").is_file()


def test_video_resources_are_released_when_inference_raises(tmp_path: Path) -> None:
    videos = _videos()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"placeholder")
    output = tmp_path / "output.mp4"

    class FakeCapture:
        released = False
        reads = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            return {
                cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48,
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 1,
            }.get(prop, 0)

        def read(self) -> tuple[bool, np.ndarray | None]:
            self.reads += 1
            if self.reads == 1:
                return True, np.zeros((48, 64, 3), dtype=np.uint8)
            return False, None

        def release(self) -> None:
            self.released = True

    class FakeWriter:
        released = False

        def isOpened(self) -> bool:
            return True

        def write(self, frame: np.ndarray) -> None:
            pass

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    writer = FakeWriter()

    class ExplodingDetector:
        def predict_bgr(self, image: np.ndarray) -> object:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        videos.run_video_inference(
            ExplodingDetector(),
            source,
            output,
            capture_factory=lambda _: capture,
            writer_factory=lambda *_: writer,
        )

    assert capture.released
    assert writer.released


def test_video_flow_rejects_existing_output_without_force(tmp_path: Path) -> None:
    videos = _videos()
    source = tmp_path / "input.mp4"
    source.write_bytes(b"input")
    output = tmp_path / "output.mp4"
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--force"):
        videos.run_video_inference(FakeDetector(), source, output)
