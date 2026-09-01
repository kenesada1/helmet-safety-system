"""FastAPI application for versioned OpenCV/ONNX helmet inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
from threading import Lock
from typing import Annotated, Callable, Iterator

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel

from helmet_safety.inference.opencv import OpenCVDetector
from helmet_safety.service.monitoring import InferenceMonitor
from helmet_safety.service.registry import load_model_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs" / "model_registry.json"
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
DetectorFactory = Callable[[], object]


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    registry_path: Path = field(default_factory=lambda: DEFAULT_REGISTRY)
    model_id: str = "e4-yolo11s-960-onnx"
    device: str = "cpu"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    confidence_baseline: float = 0.80
    confidence_drift_threshold: float = 0.15
    metrics_window_size: int = 200
    verify_artifact: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "registry_path", self.registry_path.expanduser().resolve())
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty")
        if self.max_upload_bytes < 1 or self.max_image_pixels < 1:
            raise ValueError("upload and image limits must be positive")
        if self.metrics_window_size < 1:
            raise ValueError("metrics_window_size must be positive")

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        return cls(
            registry_path=Path(os.getenv("HELMET_MODEL_REGISTRY", str(DEFAULT_REGISTRY))),
            model_id=os.getenv("HELMET_MODEL_ID", "e4-yolo11s-960-onnx"),
            device=os.getenv("HELMET_DEVICE", "cpu"),
            max_upload_bytes=int(os.getenv("HELMET_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            max_image_pixels=int(os.getenv("HELMET_MAX_IMAGE_PIXELS", "20000000")),
            confidence_baseline=float(os.getenv("HELMET_CONFIDENCE_BASELINE", "0.80")),
            confidence_drift_threshold=float(
                os.getenv("HELMET_CONFIDENCE_DRIFT_THRESHOLD", "0.15")
            ),
            metrics_window_size=int(os.getenv("HELMET_METRICS_WINDOW_SIZE", "200")),
            verify_artifact=os.getenv("HELMET_VERIFY_ARTIFACT", "1").lower()
            not in {"0", "false", "no"},
        )


class ImageShape(BaseModel):
    width: int
    height: int


class DetectionPayload(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


class DetectionResponse(BaseModel):
    model_id: str
    image: ImageShape
    inference_ms: float
    detections: list[DetectionPayload]


class LiveResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    model_id: str


def _default_detector_factory(settings: ServiceSettings) -> DetectorFactory:
    def build() -> OpenCVDetector:
        artifact = load_model_artifact(
            settings.registry_path,
            settings.model_id,
            verify=settings.verify_artifact,
        )
        return OpenCVDetector(
            weights=artifact.artifact_path,
            device=settings.device,
            imgsz=artifact.imgsz,
            conf=0.25,
            iou=0.70,
            max_det=300,
        )

    return build


def _decode_upload(image: UploadFile, settings: ServiceSettings) -> np.ndarray:
    if not image.content_type or not image.content_type.lower().startswith("image/"):
        raise HTTPException(status_code=415, detail="upload must have an image media type")
    content = image.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="uploaded image is too large")
    if not content:
        raise HTTPException(status_code=422, detail="uploaded image is empty")
    decoded = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise HTTPException(status_code=422, detail="uploaded content is not a decodable image")
    height, width = decoded.shape[:2]
    if height * width > settings.max_image_pixels:
        raise HTTPException(status_code=413, detail="decoded image has too many pixels")
    return decoded


def create_app(
    *,
    detector: object | None = None,
    detector_factory: DetectorFactory | None = None,
    model_id: str | None = None,
    settings: ServiceSettings | None = None,
) -> FastAPI:
    if detector is not None and detector_factory is not None:
        raise ValueError("provide either detector or detector_factory, not both")
    settings = settings or ServiceSettings.from_env()
    active_model_id = model_id or settings.model_id
    factory = detector_factory or _default_detector_factory(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        if app.state.detector is None:
            try:
                app.state.detector = factory()
                app.state.load_error = None
            except Exception as exc:
                app.state.load_error = exc
        yield
        app.state.detector = None

    app = FastAPI(
        title="Helmet Safety Inference API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.detector = detector
    app.state.load_error = None
    app.state.inference_lock = Lock()
    app.state.monitor = InferenceMonitor(
        confidence_baseline=settings.confidence_baseline,
        confidence_drift_threshold=settings.confidence_drift_threshold,
        window_size=settings.metrics_window_size,
    )

    @app.get("/health/live", tags=["health"])
    def live() -> LiveResponse:
        return LiveResponse(status="ok")

    @app.get("/health/ready", tags=["health"])
    def ready() -> ReadyResponse:
        if app.state.detector is None:
            raise HTTPException(status_code=503, detail="model is not ready")
        return ReadyResponse(status="ready", model_id=active_model_id)

    @app.post("/v1/detections", tags=["inference"])
    def detect(
        image: Annotated[UploadFile, File(description="JPEG, PNG, or BMP image")],
    ) -> DetectionResponse:
        if app.state.detector is None:
            raise HTTPException(status_code=503, detail="model is not ready")
        frame = _decode_upload(image, settings)
        try:
            with app.state.inference_lock:
                result = app.state.detector.predict_bgr(frame)
            app.state.monitor.record_success(
                inference_seconds=result.inference_seconds,
                detections=result.detections,
            )
        except Exception as exc:
            app.state.monitor.record_failure()
            raise HTTPException(status_code=500, detail="inference failed") from exc
        height, width = frame.shape[:2]
        return DetectionResponse(
            model_id=active_model_id,
            image=ImageShape(width=width, height=height),
            inference_ms=result.inference_seconds * 1000.0,
            detections=[DetectionPayload(**item.to_dict()) for item in result.detections],
        )

    @app.get("/v1/metrics", tags=["observability"])
    def metrics_json() -> dict[str, object]:
        return app.state.monitor.snapshot()

    @app.get("/metrics", tags=["observability"], response_class=PlainTextResponse)
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(
            app.state.monitor.render_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/", include_in_schema=False)
    def monitor_page() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")

    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend-assets")

    return app


app = create_app()
