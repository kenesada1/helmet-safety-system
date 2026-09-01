from __future__ import annotations

import math
from typing import Mapping, Sequence


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _equivalent_size(box: Sequence[float]) -> float:
    return math.sqrt(max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1])))


def _is_edge(box: Sequence[float], image_size: tuple[int, int]) -> bool:
    width, height = image_size
    center_x = (float(box[0]) + float(box[2])) / 2
    center_y = (float(box[1]) + float(box[3])) / 2
    return center_x <= width * 0.1 or center_x >= width * 0.9 or center_y <= height * 0.1 or center_y >= height * 0.9


def _match_pairs(
    ground_truth: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    matching_iou: float,
) -> list[tuple[int, int, float]]:
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
    matches: list[tuple[int, int, float]] = []
    for iou, gt_index, prediction_index in candidates:
        if iou < matching_iou:
            break
        if gt_index in matched_gt or prediction_index in matched_predictions:
            continue
        matched_gt.add(gt_index)
        matched_predictions.add(prediction_index)
        matches.append((gt_index, prediction_index, iou))
    return matches


def analyze_model_errors(
    ground_truth: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    image_size: tuple[int, int],
    confidence_threshold: float = 0.25,
    matching_iou: float = 0.5,
) -> dict[str, object]:
    fixed_predictions = [
        (index, prediction)
        for index, prediction in enumerate(predictions)
        if float(prediction["confidence"]) >= confidence_threshold
    ]
    fixed_only = [prediction for _, prediction in fixed_predictions]
    matches = _match_pairs(ground_truth, fixed_only, matching_iou=matching_iou)
    matched_gt = {gt_index for gt_index, _, _ in matches}
    matched_fixed_predictions = {prediction_index for _, prediction_index, _ in matches}

    false_negatives: list[dict[str, object]] = []
    for gt_index, gt in enumerate(ground_truth):
        if gt_index in matched_gt:
            continue
        same_class = [
            (index, prediction, _box_iou(gt["box"], prediction["box"]))  # type: ignore[arg-type]
            for index, prediction in enumerate(predictions)
            if int(prediction["class_id"]) == int(gt["class_id"])
        ]
        wrong_class = [
            (index, prediction, _box_iou(gt["box"], prediction["box"]))  # type: ignore[arg-type]
            for index, prediction in enumerate(predictions)
            if int(prediction["class_id"]) != int(gt["class_id"])
        ]
        best_same = max(same_class, key=lambda item: item[2], default=None)
        best_wrong = max(wrong_class, key=lambda item: item[2], default=None)
        same_iou = float(best_same[2]) if best_same else 0.0
        same_confidence = float(best_same[1]["confidence"]) if best_same else None
        wrong_iou = float(best_wrong[2]) if best_wrong else 0.0
        wrong_confidence = float(best_wrong[1]["confidence"]) if best_wrong else None
        if wrong_iou >= matching_iou and wrong_confidence is not None and wrong_confidence >= confidence_threshold:
            reason = "class_confusion"
        elif same_iou >= matching_iou and same_confidence is not None and same_confidence < confidence_threshold:
            reason = "low_confidence"
        elif same_iou >= 0.1:
            reason = "localization"
        else:
            reason = "no_response"
        box = [float(value) for value in gt["box"]]  # type: ignore[union-attr]
        false_negatives.append(
            {
                "gt_index": gt_index,
                "class_id": int(gt["class_id"]),
                "box": box,
                "reason": reason,
                "same_class_max_iou": same_iou,
                "same_class_confidence": same_confidence,
                "wrong_class_max_iou": wrong_iou,
                "wrong_class_confidence": wrong_confidence,
                "equivalent_size": _equivalent_size(box),
                "edge": _is_edge(box, image_size),
            }
        )

    false_positives: list[dict[str, object]] = []
    for fixed_index, (original_index, prediction) in enumerate(fixed_predictions):
        if fixed_index in matched_fixed_predictions:
            continue
        overlaps = [
            (gt_index, gt, _box_iou(gt["box"], prediction["box"]))  # type: ignore[arg-type]
            for gt_index, gt in enumerate(ground_truth)
        ]
        nearest = max(overlaps, key=lambda item: item[2], default=None)
        max_iou = float(nearest[2]) if nearest else 0.0
        nearest_class_id = int(nearest[1]["class_id"]) if nearest and max_iou > 0 else None
        if max_iou >= matching_iou and nearest_class_id != int(prediction["class_id"]):
            reason = "class_confusion"
        elif max_iou >= matching_iou:
            reason = "duplicate"
        elif max_iou >= 0.1:
            reason = "near_object"
        else:
            reason = "background"
        box = [float(value) for value in prediction["box"]]  # type: ignore[union-attr]
        false_positives.append(
            {
                "prediction_index": original_index,
                "class_id": int(prediction["class_id"]),
                "confidence": float(prediction["confidence"]),
                "box": box,
                "reason": reason,
                "max_iou": max_iou,
                "nearest_gt_class_id": nearest_class_id,
                "equivalent_size": _equivalent_size(box),
                "edge": _is_edge(box, image_size),
            }
        )
    return {
        "matches": matches,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def compare_gt_outcomes(
    ground_truth: Sequence[Mapping[str, object]],
    e4_predictions: Sequence[Mapping[str, object]],
    e6_predictions: Sequence[Mapping[str, object]],
    *,
    confidence_threshold: float = 0.25,
    matching_iou: float = 0.5,
) -> list[dict[str, object]]:
    def matched_indices(predictions: Sequence[Mapping[str, object]]) -> set[int]:
        fixed = [prediction for prediction in predictions if float(prediction["confidence"]) >= confidence_threshold]
        return {gt_index for gt_index, _, _ in _match_pairs(ground_truth, fixed, matching_iou=matching_iou)}

    e4_matched = matched_indices(e4_predictions)
    e6_matched = matched_indices(e6_predictions)
    transitions: list[dict[str, object]] = []
    for gt_index, gt in enumerate(ground_truth):
        e4_hit = gt_index in e4_matched
        e6_hit = gt_index in e6_matched
        if e4_hit and e6_hit:
            transition = "both_detected"
        elif not e4_hit and e6_hit:
            transition = "recovered"
        elif e4_hit and not e6_hit:
            transition = "regressed"
        else:
            transition = "both_missed"
        transitions.append(
            {
                "gt_index": gt_index,
                "class_id": int(gt["class_id"]),
                "box": [float(value) for value in gt["box"]],  # type: ignore[union-attr]
                "equivalent_size": _equivalent_size(gt["box"]),  # type: ignore[arg-type]
                "transition": transition,
            }
        )
    return transitions
