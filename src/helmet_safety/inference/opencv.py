"""Reusable Ultralytics inference and OpenCV rendering for M5."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np


CLASS_NAMES: dict[int, str] = {0: "helmet", 1: "no_helmet"}
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "helmet": (0, 200, 0),
    "no_helmet": (0, 0, 255),
}


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "xyxy": list(self.xyxy),
        }


@dataclass(frozen=True, slots=True)
class InferenceResult:
    detections: list[Detection]
    inference_seconds: float


def _to_numpy(value: object) -> np.ndarray:
    detached = value.detach() if hasattr(value, "detach") else value
    on_cpu = detached.cpu() if hasattr(detached, "cpu") else detached
    array = on_cpu.numpy() if hasattr(on_cpu, "numpy") else on_cpu
    return np.asarray(array)


def detections_from_ultralytics(result: object) -> list[Detection]:
    """Convert one Ultralytics Results object into stable project records."""

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []
    class_ids = _to_numpy(boxes.cls).reshape(-1)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    coordinates = _to_numpy(boxes.xyxy).reshape(-1, 4)
    if not (len(class_ids) == len(confidences) == len(coordinates)):
        raise RuntimeError("Ultralytics output has inconsistent box arrays")
    result_names = getattr(result, "names", CLASS_NAMES)
    names: Mapping[int, str] = {
        int(key): str(value) for key, value in dict(result_names).items()
    }
    converted: list[Detection] = []
    for raw_class_id, raw_confidence, raw_xyxy in zip(
        class_ids, confidences, coordinates, strict=True
    ):
        class_id = int(raw_class_id)
        class_name = names.get(class_id, CLASS_NAMES.get(class_id))
        if class_name not in CLASS_COLORS:
            raise ValueError(f"unsupported detection class {class_id}: {class_name!r}")
        converted.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(raw_confidence),
                xyxy=tuple(float(value) for value in raw_xyxy),  # type: ignore[arg-type]
            )
        )
    return converted


def _validate_model_names(model: object) -> None:
    raw_names = getattr(model, "names", None)
    if raw_names is None:
        return
    names = {int(key): str(value) for key, value in dict(raw_names).items()}
    if names != CLASS_NAMES:
        raise ValueError(
            f"weights class mapping must be {CLASS_NAMES}, received {names}"
        )


class OpenCVDetector:
    """Load a detection model once and run it against OpenCV BGR frames."""

    def __init__(
        self,
        *,
        weights: Path | str,
        device: str = "0",
        imgsz: int = 960,
        conf: float = 0.25,
        iou: float = 0.70,
        max_det: int = 300,
        fp16: bool = False,
        model: object | None = None,
        model_factory: Callable[[str], object] | None = None,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"model weights do not exist: {self.weights}")
        if imgsz < 1:
            raise ValueError("imgsz must be positive")
        if not 0.0 <= conf <= 1.0:
            raise ValueError("conf must be within [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be within [0, 1]")
        if max_det < 1:
            raise ValueError("max_det must be positive")
        if model is not None and model_factory is not None:
            raise ValueError("provide either model or model_factory, not both")
        self.device = str(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.fp16 = bool(fp16)
        if model is None:
            try:
                if model_factory is None:
                    from ultralytics import YOLO

                    model = YOLO(str(self.weights), task="detect")
                else:
                    model = model_factory(str(self.weights))
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load model weights {self.weights}: {exc}"
                ) from exc
        _validate_model_names(model)
        if self.fp16 and hasattr(model, "half"):
            model = model.half()
        self._model = model

    def predict_bgr(self, image: np.ndarray) -> InferenceResult:
        if (
            not isinstance(image, np.ndarray)
            or image.dtype != np.uint8
            or image.ndim != 3
            or image.shape[2] != 3
            or image.shape[0] < 1
            or image.shape[1] < 1
        ):
            raise ValueError("input must be a non-empty uint8 OpenCV BGR image")
        started = perf_counter()
        raw_results = self._model.predict(
            source=image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            verbose=False,
        )
        inference_seconds = perf_counter() - started
        if len(raw_results) != 1:
            raise RuntimeError(
                f"expected one Ultralytics result for one BGR image, got {len(raw_results)}"
            )
        return InferenceResult(
            detections=detections_from_ultralytics(raw_results[0]),
            inference_seconds=inference_seconds,
        )


def _clipped_box(
    xyxy: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    left, right = sorted((float(xyxy[0]), float(xyxy[2])))
    top, bottom = sorted((float(xyxy[1]), float(xyxy[3])))
    x1 = min(max(int(round(left)), 0), width - 1)
    y1 = min(max(int(round(top)), 0), height - 1)
    x2 = min(max(int(round(right)), 0), width - 1)
    y2 = min(max(int(round(bottom)), 0), height - 1)
    if x2 == x1 and x2 < width - 1:
        x2 += 1
    if y2 == y1 and y2 < height - 1:
        y2 += 1
    return x1, y1, x2, y2


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    *,
    processing_fps: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Return an annotated copy plus per-frame detection-box counts."""

    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or image.shape[0] < 1
        or image.shape[1] < 1
    ):
        raise ValueError("drawing input must be a non-empty uint8 BGR image")
    output = image.copy()
    height, width = output.shape[:2]
    thickness = max(1, round(min(width, height) / 500))
    font_scale = max(0.35, min(width, height) / 700)
    counts = {"helmet": 0, "no_helmet": 0}
    for detection in detections:
        if detection.class_name not in CLASS_COLORS:
            raise ValueError(f"unsupported detection class: {detection.class_name}")
        counts[detection.class_name] += 1
        x1, y1, x2, y2 = _clipped_box(detection.xyxy, width, height)
        color = CLASS_COLORS[detection.class_name]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness, cv2.LINE_8)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        label_right = min(width - 1, x1 + text_width + 6)
        cv2.rectangle(
            output,
            (x1, label_top),
            (label_right, min(height - 1, label_top + text_height + baseline + 4)),
            color,
            -1,
        )
        text_y = min(height - 1, label_top + text_height + 2)
        cv2.putText(
            output,
            label,
            (min(width - 1, x1 + 3), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    panel_lines = (
        f"helmet: {counts['helmet']}",
        f"no_helmet: {counts['no_helmet']}",
        f"FPS: {max(0.0, processing_fps):.1f}",
    )
    panel_scale = max(0.38, min(width, height) / 650)
    panel_thickness = max(1, thickness)
    sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, panel_scale, panel_thickness)[0]
        for line in panel_lines
    ]
    line_height = max(size[1] for size in sizes) + 7
    panel_width = min(width, max(size[0] for size in sizes) + 16)
    panel_height = min(height, line_height * len(panel_lines) + 8)
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (panel_width - 1, panel_height - 1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, output, 0.28, 0, output)
    for index, line in enumerate(panel_lines):
        y = min(height - 1, 6 + (index + 1) * line_height - 5)
        cv2.putText(
            output,
            line,
            (7, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            panel_scale,
            (255, 255, 255),
            panel_thickness,
            cv2.LINE_AA,
        )
    return output, counts
