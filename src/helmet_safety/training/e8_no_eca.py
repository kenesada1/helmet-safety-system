from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch
from torch import nn

from helmet_safety.training.e7_lfr import (
    E4_DETECT_INDEX,
    E7_DETECT_INDEX,
    E7_LFR_INDEX,
    PHYSICAL_VRAM_UTILIZATION_LIMIT,
    assert_output_available,
    build_training_kwargs,
    configure_probe_hyperparameters,
    fits_physical_vram,
    transfer_e4_state,
)


E8_REFINEMENT_INDEX = E7_LFR_INDEX
E8_DETECT_INDEX = E7_DETECT_INDEX
SIZE_BUCKETS = ("<=10", "10-20", "20-30", "30-50", ">50")


class NoECAFeatureRefinement(nn.Module):
    """E7 P2 refinement with only its ECA operation removed."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.activation(self.depthwise_bn(self.depthwise(x)))
        refined = self.pointwise_bn(self.pointwise(refined))
        return x + refined


def register_no_eca_module() -> None:
    """Expose the project-local E8 layer without modifying Ultralytics."""

    import ultralytics.nn.tasks as tasks

    tasks.__dict__[NoECAFeatureRefinement.__name__] = NoECAFeatureRefinement


def equivalent_size_bucket(size: float) -> str:
    """Return the required bucket for a box equivalent side length in pixels."""

    if not math.isfinite(size) or size < 0:
        raise ValueError("equivalent size must be a finite non-negative value")
    if size <= 10:
        return "<=10"
    if size <= 20:
        return "10-20"
    if size <= 30:
        return "20-30"
    if size <= 50:
        return "30-50"
    return ">50"


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _equivalent_box_size(box: Sequence[float]) -> float:
    return math.sqrt(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))


def summarize_size_buckets(
    records: Sequence[Mapping[str, object]], *, matching_iou: float
) -> dict[str, object]:
    """Summarize class-aware fixed-threshold outcomes by original box size."""

    if not 0 < matching_iou <= 1:
        raise ValueError("matching_iou must be in (0, 1]")
    scopes = ("overall", "helmet", "no_helmet")
    counts = {
        scope: {
            bucket: {"ground_truth": 0, "tp": 0, "fn": 0, "recall": 0.0}
            for bucket in SIZE_BUCKETS
        }
        for scope in scopes
    }
    false_positives = {bucket: 0 for bucket in SIZE_BUCKETS}
    for record in records:
        ground_truth = record["ground_truth"]  # type: ignore[assignment]
        predictions = record["predictions"]  # type: ignore[assignment]
        candidates = sorted(
            (
                (_box_iou(gt["box"], prediction["box"]), gt_index, prediction_index)  # type: ignore[arg-type]
                for gt_index, gt in enumerate(ground_truth)
                for prediction_index, prediction in enumerate(predictions)
                if int(gt["class_id"]) == int(prediction["class_id"])
            ),
            reverse=True,
        )
        matched_gt: set[int] = set()
        matched_predictions: set[int] = set()
        for overlap, gt_index, prediction_index in candidates:
            if overlap < matching_iou:
                break
            if gt_index in matched_gt or prediction_index in matched_predictions:
                continue
            matched_gt.add(gt_index)
            matched_predictions.add(prediction_index)

        for gt_index, gt in enumerate(ground_truth):
            class_scope = "helmet" if int(gt["class_id"]) == 0 else "no_helmet"
            bucket = equivalent_size_bucket(_equivalent_box_size(gt["box"]))  # type: ignore[arg-type]
            for scope in ("overall", class_scope):
                row = counts[scope][bucket]
                row["ground_truth"] += 1
                if gt_index in matched_gt:
                    row["tp"] += 1
                else:
                    row["fn"] += 1
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index not in matched_predictions:
                bucket = equivalent_size_bucket(_equivalent_box_size(prediction["box"]))  # type: ignore[arg-type]
                false_positives[bucket] += 1

    for scope in scopes:
        for bucket in SIZE_BUCKETS:
            row = counts[scope][bucket]
            row["recall"] = row["tp"] / row["ground_truth"] if row["ground_truth"] else 0.0
    return {**counts, "false_positives_by_prediction_size": false_positives}


__all__ = [
    "E4_DETECT_INDEX",
    "E8_DETECT_INDEX",
    "E8_REFINEMENT_INDEX",
    "NoECAFeatureRefinement",
    "PHYSICAL_VRAM_UTILIZATION_LIMIT",
    "SIZE_BUCKETS",
    "assert_output_available",
    "build_training_kwargs",
    "configure_probe_hyperparameters",
    "equivalent_size_bucket",
    "fits_physical_vram",
    "register_no_eca_module",
    "summarize_size_buckets",
    "transfer_e4_state",
]
