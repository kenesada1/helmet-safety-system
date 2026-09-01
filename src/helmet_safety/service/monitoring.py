"""Thread-safe, dependency-light inference telemetry and drift monitoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from math import isfinite
from threading import Lock
from typing import Any

from helmet_safety.inference.opencv import Detection


GpuStatsProvider = Callable[[], dict[str, int | float | None]]


def default_gpu_stats() -> dict[str, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_bytes": 0, "reserved_bytes": 0}
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }
    except (ImportError, RuntimeError):
        return {"allocated_bytes": None, "reserved_bytes": None}


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _number(value: float | int | None) -> str:
    if value is None:
        return "NaN"
    if isinstance(value, int):
        return str(value)
    return format(value, ".12g")


class InferenceMonitor:
    def __init__(
        self,
        *,
        confidence_baseline: float = 0.80,
        confidence_drift_threshold: float = 0.15,
        window_size: int = 200,
        gpu_stats_provider: GpuStatsProvider = default_gpu_stats,
    ) -> None:
        if not 0.0 <= confidence_baseline <= 1.0:
            raise ValueError("confidence_baseline must be within [0, 1]")
        if not 0.0 <= confidence_drift_threshold <= 1.0:
            raise ValueError("confidence_drift_threshold must be within [0, 1]")
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self._baseline = float(confidence_baseline)
        self._drift_threshold = float(confidence_drift_threshold)
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._confidences: deque[float] = deque(maxlen=window_size)
        self._success = 0
        self._errors = 0
        self._detections = {"helmet": 0, "no_helmet": 0}
        self._gpu_stats_provider = gpu_stats_provider
        self._lock = Lock()

    def record_success(
        self, *, inference_seconds: float, detections: Sequence[Detection]
    ) -> None:
        if not isfinite(inference_seconds) or inference_seconds < 0:
            raise ValueError("inference_seconds must be finite and non-negative")
        with self._lock:
            self._success += 1
            self._latencies.append(float(inference_seconds))
            for detection in detections:
                if detection.class_name not in self._detections:
                    raise ValueError(f"unsupported detection class {detection.class_name!r}")
                self._detections[detection.class_name] += 1
                self._confidences.append(float(detection.confidence))

    def record_failure(self) -> None:
        with self._lock:
            self._errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            success = self._success
            errors = self._errors
            latencies = list(self._latencies)
            confidences = list(self._confidences)
            detections = dict(self._detections)
        latency_ms = [value * 1000.0 for value in latencies]
        rolling_mean = sum(confidences) / len(confidences) if confidences else None
        drift = rolling_mean - self._baseline if rolling_mean is not None else None
        try:
            gpu_memory = self._gpu_stats_provider()
        except Exception:
            gpu_memory = {"allocated_bytes": None, "reserved_bytes": None}
        return {
            "requests": {
                "total": success + errors,
                "success": success,
                "error": errors,
            },
            "latency_ms": {
                "count": len(latency_ms),
                "mean": sum(latency_ms) / len(latency_ms) if latency_ms else 0.0,
                "p50": _percentile(latency_ms, 0.50),
                "p95": _percentile(latency_ms, 0.95),
            },
            "detections": detections,
            "confidence": {
                "samples": len(confidences),
                "baseline": self._baseline,
                "rolling_mean": rolling_mean,
                "drift": drift,
                "drift_alert": bool(
                    drift is not None and abs(drift) >= self._drift_threshold
                ),
            },
            "gpu_memory": gpu_memory,
        }

    def render_prometheus(self) -> str:
        snapshot = self.snapshot()
        requests = snapshot["requests"]
        latency = snapshot["latency_ms"]
        detections = snapshot["detections"]
        confidence = snapshot["confidence"]
        gpu = snapshot["gpu_memory"]
        lines = [
            "# HELP helmet_inference_requests_total Inference requests by outcome.",
            "# TYPE helmet_inference_requests_total counter",
            f'helmet_inference_requests_total{{status="success"}} {requests["success"]}',
            f'helmet_inference_requests_total{{status="error"}} {requests["error"]}',
            "# HELP helmet_inference_latency_seconds Rolling inference latency.",
            "# TYPE helmet_inference_latency_seconds gauge",
            f'helmet_inference_latency_seconds{{quantile="0.50"}} {_number(latency["p50"] / 1000.0)}',
            f'helmet_inference_latency_seconds{{quantile="0.95"}} {_number(latency["p95"] / 1000.0)}',
            "# HELP helmet_detections_total Detection boxes by class.",
            "# TYPE helmet_detections_total counter",
            f'helmet_detections_total{{class_name="helmet"}} {detections["helmet"]}',
            f'helmet_detections_total{{class_name="no_helmet"}} {detections["no_helmet"]}',
            "# HELP helmet_confidence_rolling_mean Rolling mean confidence.",
            "# TYPE helmet_confidence_rolling_mean gauge",
            f'helmet_confidence_rolling_mean {_number(confidence["rolling_mean"])}',
            "# HELP helmet_confidence_drift_alert Confidence drift threshold state.",
            "# TYPE helmet_confidence_drift_alert gauge",
            f'helmet_confidence_drift_alert {int(confidence["drift_alert"])}',
            "# HELP helmet_gpu_memory_bytes Current PyTorch GPU memory.",
            "# TYPE helmet_gpu_memory_bytes gauge",
            f'helmet_gpu_memory_bytes{{kind="allocated"}} {_number(gpu.get("allocated_bytes"))}',
            f'helmet_gpu_memory_bytes{{kind="reserved"}} {_number(gpu.get("reserved_bytes"))}',
        ]
        return "\n".join(lines) + "\n"

