from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from helmet_safety.inference.opencv import Detection, InferenceResult


def _service() -> object:
    try:
        from helmet_safety.service import api as module
    except ModuleNotFoundError:
        pytest.fail("helmet_safety.service.api must implement the HTTP inference service")
    return module


class FakeDetector:
    def predict_bgr(self, image: np.ndarray) -> InferenceResult:
        assert image.shape == (24, 32, 3)
        return InferenceResult(
            detections=[
                Detection(0, "helmet", 0.9, (1.0, 2.0, 12.0, 14.0)),
                Detection(1, "no_helmet", 0.6, (15.0, 3.0, 28.0, 20.0)),
            ],
            inference_seconds=0.0125,
        )


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((24, 32, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_detection_endpoint_returns_stable_project_records_and_metrics() -> None:
    service = _service()
    app = service.create_app(detector=FakeDetector(), model_id="test-model")

    with TestClient(app) as client:
        response = client.post(
            "/v1/detections",
            files={"image": ("sample.jpg", _jpeg_bytes(), "image/jpeg")},
        )
        metrics = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_id"] == "test-model"
    assert payload["image"] == {"width": 32, "height": 24}
    assert payload["inference_ms"] == pytest.approx(12.5)
    assert payload["detections"] == [
        {
            "class_id": 0,
            "class_name": "helmet",
            "confidence": pytest.approx(0.9),
            "xyxy": [1.0, 2.0, 12.0, 14.0],
        },
        {
            "class_id": 1,
            "class_name": "no_helmet",
            "confidence": pytest.approx(0.6),
            "xyxy": [15.0, 3.0, 28.0, 20.0],
        },
    ]
    assert metrics.status_code == 200
    assert 'helmet_inference_requests_total{status="success"} 1' in metrics.text
    assert 'helmet_detections_total{class_name="helmet"} 1' in metrics.text
    assert 'helmet_detections_total{class_name="no_helmet"} 1' in metrics.text


def test_health_endpoints_distinguish_liveness_and_model_readiness() -> None:
    service = _service()
    ready_app = service.create_app(detector=FakeDetector(), model_id="test-model")
    unavailable_app = service.create_app(
        detector_factory=lambda: (_ for _ in ()).throw(RuntimeError("missing model")),
        model_id="missing-model",
    )

    with TestClient(ready_app) as client:
        assert client.get("/health/live").json() == {"status": "ok"}
        ready = client.get("/health/ready")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready", "model_id": "test-model"}

    with TestClient(unavailable_app, raise_server_exceptions=False) as client:
        assert client.get("/health/live").status_code == 200
        not_ready = client.get("/health/ready")
        assert not_ready.status_code == 503
        assert not_ready.json()["detail"] == "model is not ready"


def test_detection_endpoint_rejects_unsafe_or_invalid_uploads() -> None:
    service = _service()
    settings = service.ServiceSettings(max_upload_bytes=32, max_image_pixels=10_000)
    app = service.create_app(
        detector=FakeDetector(), model_id="test-model", settings=settings
    )

    with TestClient(app) as client:
        wrong_type = client.post(
            "/v1/detections",
            files={"image": ("sample.txt", b"not an image", "text/plain")},
        )
        too_large = client.post(
            "/v1/detections",
            files={"image": ("sample.jpg", b"x" * 33, "image/jpeg")},
        )
        malformed = client.post(
            "/v1/detections",
            files={"image": ("sample.jpg", b"bad", "image/jpeg")},
        )

    assert wrong_type.status_code == 415
    assert too_large.status_code == 413
    assert malformed.status_code == 422


def test_service_settings_load_environment_without_hard_coded_machine_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    service = _service()
    registry = tmp_path / "registry.json"
    monkeypatch.setenv("HELMET_MODEL_REGISTRY", str(registry))
    monkeypatch.setenv("HELMET_MODEL_ID", "e4-onnx")
    monkeypatch.setenv("HELMET_DEVICE", "cpu")
    monkeypatch.setenv("HELMET_MAX_UPLOAD_BYTES", "2048")

    settings = service.ServiceSettings.from_env()

    assert settings.registry_path == registry.resolve()
    assert settings.model_id == "e4-onnx"
    assert settings.device == "cpu"
    assert settings.max_upload_bytes == 2048


def test_monitor_page_and_browser_assets_are_served_from_the_api_origin() -> None:
    service = _service()
    app = service.create_app(detector=FakeDetector(), model_id="test-model")

    with TestClient(app) as client:
        page = client.get("/")
        script = client.get("/assets/app.js")
        stylesheet = client.get("/assets/styles.css")

    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "头盔安全实时监控" in page.text
    assert 'id="camera-select"' in page.text
    assert 'id="monitor-canvas"' in page.text
    assert 'id="start-monitor"' in page.text
    assert script.status_code == 200
    assert "navigator.mediaDevices.getUserMedia" in script.text
    assert 'fetch("/v1/detections"' in script.text
    assert stylesheet.status_code == 200
    assert stylesheet.headers["content-type"].startswith("text/css")
