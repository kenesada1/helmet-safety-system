from __future__ import annotations

from decimal import Decimal
from typing import Mapping, Sequence

from helmet_safety.training.analysis_core import (
    summarize_fixed_threshold_detections,
    summarize_tiny_ground_truth,
)


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    """Build an inclusive decimal grid without binary floating-point drift."""

    start_decimal = Decimal(str(start))
    stop_decimal = Decimal(str(stop))
    step_decimal = Decimal(str(step))
    if step_decimal <= 0:
        raise ValueError("threshold step must be positive")
    if stop_decimal < start_decimal:
        raise ValueError("threshold stop must not be smaller than start")
    values: list[float] = []
    current = start_decimal
    while current <= stop_decimal:
        values.append(float(current))
        current += step_decimal
    return values


def apply_class_thresholds(
    records: Sequence[Mapping[str, object]], thresholds: Mapping[int, float]
) -> list[dict[str, object]]:
    """Filter post-NMS predictions with an independent confidence per class."""

    if set(thresholds) != {0, 1}:
        raise ValueError("thresholds must contain exactly class ids 0 and 1")
    normalized = {class_id: float(value) for class_id, value in thresholds.items()}
    if any(not 0.0 <= value <= 1.0 for value in normalized.values()):
        raise ValueError("confidence thresholds must be within [0, 1]")

    filtered: list[dict[str, object]] = []
    for record in records:
        predictions = [
            dict(prediction)
            for prediction in record["predictions"]  # type: ignore[union-attr]
            if float(prediction["confidence"]) >= normalized[int(prediction["class_id"])]
        ]
        filtered.append({**dict(record), "predictions": predictions})
    return filtered


def evaluate_threshold_point(
    records: Sequence[Mapping[str, object]],
    *,
    helmet_conf: float,
    no_helmet_conf: float,
    matching_iou: float = 0.5,
) -> dict[str, object]:
    """Evaluate one pair of class-specific confidence thresholds."""

    filtered = apply_class_thresholds(records, {0: helmet_conf, 1: no_helmet_conf})
    fixed = summarize_fixed_threshold_detections(filtered, iou_threshold=matching_iou)
    tiny = summarize_tiny_ground_truth(filtered, iou_threshold=matching_iou)["summary"]
    return {
        "thresholds": {"helmet": float(helmet_conf), "no_helmet": float(no_helmet_conf)},
        "overall": fixed["overall"],
        "per_class": fixed["per_class"],
        "tiny": tiny,
    }


def compose_class_points(
    *,
    helmet_conf: float,
    no_helmet_conf: float,
    helmet: Mapping[str, object],
    no_helmet: Mapping[str, object],
    helmet_tiny: Mapping[str, object],
    no_helmet_tiny: Mapping[str, object],
    images: int,
) -> dict[str, object]:
    """Compose independent class curves into one exact class-aware operating point."""

    def metric_row(counts: Mapping[str, object]) -> dict[str, object]:
        ground_truth = int(counts["ground_truth"])
        predictions = int(counts["predictions"])
        tp = int(counts["tp"])
        fn = int(counts["fn"])
        fp = int(counts["fp"])
        return {
            "ground_truth": ground_truth,
            "predictions": predictions,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "fp_per_image": fp / images if images else None,
        }

    helmet_row = metric_row(helmet)
    no_helmet_row = metric_row(no_helmet)
    overall_counts = {
        key: int(helmet_row[key]) + int(no_helmet_row[key])
        for key in ("ground_truth", "predictions", "tp", "fn", "fp")
    }
    overall = {"images": images, **metric_row(overall_counts)}
    tiny_tp = int(helmet_tiny["tp"]) + int(no_helmet_tiny["tp"])
    tiny_fn = int(helmet_tiny["fn"]) + int(no_helmet_tiny["fn"])
    tiny_ground_truth = int(helmet_tiny["ground_truth"]) + int(no_helmet_tiny["ground_truth"])
    return {
        "thresholds": {"helmet": float(helmet_conf), "no_helmet": float(no_helmet_conf)},
        "overall": overall,
        "per_class": {"helmet": helmet_row, "no_helmet": no_helmet_row},
        "tiny": {
            "ground_truth": tiny_ground_truth,
            "tp": tiny_tp,
            "fn": tiny_fn,
            "recall": tiny_tp / (tiny_tp + tiny_fn) if tiny_tp + tiny_fn else None,
        },
    }


def select_operating_point(
    points: Sequence[Mapping[str, object]],
    *,
    max_false_positives: int,
    min_no_helmet_recall: float,
    min_tiny_recall: float,
    min_overall_f1: float,
) -> dict[str, object]:
    """Choose the highest-F1 point after enforcing production constraints."""

    feasible = [
        point
        for point in points
        if int(point["overall"]["fp"]) <= max_false_positives  # type: ignore[index]
        and float(point["per_class"]["no_helmet"]["recall"]) >= min_no_helmet_recall  # type: ignore[index]
        and float(point["tiny"]["recall"]) >= min_tiny_recall  # type: ignore[index]
        and float(point["overall"]["f1"]) >= min_overall_f1  # type: ignore[index]
    ]
    if not feasible:
        raise RuntimeError("no threshold pair satisfies the production constraints")
    selected = max(
        feasible,
        key=lambda point: (
            float(point["overall"]["f1"]),  # type: ignore[index]
            float(point["per_class"]["no_helmet"]["recall"]),  # type: ignore[index]
            float(point["tiny"]["recall"]),  # type: ignore[index]
            -int(point["overall"]["fp"]),  # type: ignore[index]
        ),
    )
    return {
        **dict(selected),
        "selection": {
            "feasible_points": len(feasible),
            "objective": "maximize overall F1 after all constraints pass",
            "constraints": {
                "max_false_positives": max_false_positives,
                "min_no_helmet_recall": min_no_helmet_recall,
                "min_tiny_recall": min_tiny_recall,
                "min_overall_f1": min_overall_f1,
            },
        },
    }
