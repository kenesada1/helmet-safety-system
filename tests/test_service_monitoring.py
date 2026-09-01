from __future__ import annotations

import pytest

from helmet_safety.inference.opencv import Detection


def _monitoring() -> object:
    try:
        from helmet_safety.service import monitoring as module
    except ModuleNotFoundError:
        pytest.fail("helmet_safety.service.monitoring must expose production metrics")
    return module


def test_monitor_reports_latency_gpu_and_confidence_drift() -> None:
    monitoring = _monitoring()
    monitor = monitoring.InferenceMonitor(
        confidence_baseline=0.8,
        confidence_drift_threshold=0.15,
        window_size=4,
        gpu_stats_provider=lambda: {
            "allocated_bytes": 1024,
            "reserved_bytes": 2048,
        },
    )
    detections = [Detection(1, "no_helmet", 0.5, (1.0, 1.0, 2.0, 2.0))]

    monitor.record_success(inference_seconds=0.010, detections=detections)
    monitor.record_success(inference_seconds=0.030, detections=detections)
    monitor.record_failure()
    snapshot = monitor.snapshot()

    assert snapshot["requests"] == {"total": 3, "success": 2, "error": 1}
    assert snapshot["latency_ms"] == {
        "count": 2,
        "mean": pytest.approx(20.0),
        "p50": pytest.approx(20.0),
        "p95": pytest.approx(29.0),
    }
    assert snapshot["detections"] == {"helmet": 0, "no_helmet": 2}
    assert snapshot["confidence"]["rolling_mean"] == pytest.approx(0.5)
    assert snapshot["confidence"]["drift"] == pytest.approx(-0.3)
    assert snapshot["confidence"]["drift_alert"] is True
    assert snapshot["gpu_memory"] == {
        "allocated_bytes": 1024,
        "reserved_bytes": 2048,
    }


def test_monitor_prometheus_output_is_a_consistent_snapshot() -> None:
    monitoring = _monitoring()
    monitor = monitoring.InferenceMonitor(confidence_baseline=0.75)
    monitor.record_success(
        inference_seconds=0.02,
        detections=[Detection(0, "helmet", 0.9, (1.0, 1.0, 2.0, 2.0))],
    )

    rendered = monitor.render_prometheus()

    assert "# TYPE helmet_inference_latency_seconds gauge" in rendered
    assert 'helmet_inference_requests_total{status="success"} 1' in rendered
    assert "helmet_inference_latency_seconds{quantile=\"0.95\"} 0.02" in rendered
    assert "helmet_confidence_rolling_mean 0.9" in rendered
    assert rendered.endswith("\n")

