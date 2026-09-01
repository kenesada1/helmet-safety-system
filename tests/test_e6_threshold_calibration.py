from __future__ import annotations

import pytest


def _records() -> list[dict[str, object]]:
    return [
        {
            "image_id": "one.jpg",
            "ground_truth": [
                {"class_id": 0, "box": [0.0, 0.0, 8.0, 8.0]},
                {"class_id": 1, "box": [20.0, 0.0, 28.0, 8.0]},
            ],
            "predictions": [
                {"class_id": 0, "confidence": 0.30, "box": [0.0, 0.0, 8.0, 8.0]},
                {"class_id": 0, "confidence": 0.29, "box": [50.0, 0.0, 58.0, 8.0]},
                {"class_id": 1, "confidence": 0.20, "box": [20.0, 0.0, 28.0, 8.0]},
                {"class_id": 1, "confidence": 0.19, "box": [70.0, 0.0, 78.0, 8.0]},
            ],
        }
    ]


def test_apply_class_thresholds_filters_each_class_independently_and_inclusively() -> None:
    from helmet_safety.training.threshold_calibration import apply_class_thresholds

    filtered = apply_class_thresholds(_records(), {0: 0.30, 1: 0.20})

    assert filtered[0]["predictions"] == [
        {"class_id": 0, "confidence": 0.30, "box": [0.0, 0.0, 8.0, 8.0]},
        {"class_id": 1, "confidence": 0.20, "box": [20.0, 0.0, 28.0, 8.0]},
    ]
    with pytest.raises(ValueError, match="exactly class ids 0 and 1"):
        apply_class_thresholds(_records(), {0: 0.25})


def test_evaluate_threshold_point_reports_full_and_tiny_metrics() -> None:
    from helmet_safety.training.threshold_calibration import evaluate_threshold_point

    point = evaluate_threshold_point(
        _records(), helmet_conf=0.30, no_helmet_conf=0.21, matching_iou=0.50
    )

    assert point["thresholds"] == {"helmet": 0.30, "no_helmet": 0.21}
    assert point["overall"]["tp"] == 1
    assert point["overall"]["fn"] == 1
    assert point["overall"]["fp"] == 0
    assert point["per_class"]["helmet"]["recall"] == 1.0
    assert point["per_class"]["no_helmet"]["recall"] == 0.0
    assert point["tiny"]["tp"] == 1
    assert point["tiny"]["fn"] == 1


def test_select_operating_point_enforces_constraints_before_maximizing_f1() -> None:
    from helmet_safety.training.threshold_calibration import select_operating_point

    points = [
        {
            "thresholds": {"helmet": 0.30, "no_helmet": 0.20},
            "overall": {"fp": 90, "f1": 0.91},
            "per_class": {"no_helmet": {"recall": 0.95}},
            "tiny": {"recall": 0.65},
        },
        {
            "thresholds": {"helmet": 0.31, "no_helmet": 0.19},
            "overall": {"fp": 101, "f1": 0.93},
            "per_class": {"no_helmet": {"recall": 0.96}},
            "tiny": {"recall": 0.67},
        },
        {
            "thresholds": {"helmet": 0.32, "no_helmet": 0.18},
            "overall": {"fp": 95, "f1": 0.92},
            "per_class": {"no_helmet": {"recall": 0.94}},
            "tiny": {"recall": 0.66},
        },
    ]

    selected = select_operating_point(
        points,
        max_false_positives=100,
        min_no_helmet_recall=0.945,
        min_tiny_recall=0.64,
        min_overall_f1=0.90,
    )

    assert selected["thresholds"] == {"helmet": 0.30, "no_helmet": 0.20}
    assert selected["selection"]["feasible_points"] == 1


def test_threshold_values_include_decimal_endpoints_without_float_drift() -> None:
    from helmet_safety.training.threshold_calibration import threshold_values

    assert threshold_values(0.15, 0.18, 0.01) == [0.15, 0.16, 0.17, 0.18]
    with pytest.raises(ValueError, match="positive"):
        threshold_values(0.15, 0.18, 0.0)


def test_compose_class_points_sums_counts_and_recomputes_overall_metrics() -> None:
    from helmet_safety.training.threshold_calibration import compose_class_points

    point = compose_class_points(
        helmet_conf=0.31,
        no_helmet_conf=0.21,
        helmet={"ground_truth": 10, "predictions": 10, "tp": 8, "fn": 2, "fp": 2},
        no_helmet={"ground_truth": 20, "predictions": 20, "tp": 18, "fn": 2, "fp": 2},
        helmet_tiny={"ground_truth": 2, "tp": 1, "fn": 1},
        no_helmet_tiny={"ground_truth": 8, "tp": 6, "fn": 2},
        images=3,
    )

    assert point["overall"]["tp"] == 26
    assert point["overall"]["fn"] == 4
    assert point["overall"]["fp"] == 4
    assert point["overall"]["precision"] == pytest.approx(26 / 30)
    assert point["overall"]["recall"] == pytest.approx(26 / 30)
    assert point["overall"]["f1"] == pytest.approx(26 / 30)
    assert point["tiny"] == {"ground_truth": 10, "tp": 7, "fn": 3, "recall": 0.7}
