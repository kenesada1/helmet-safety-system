from __future__ import annotations

from helmet_safety.training.e6_error_analysis import analyze_model_errors, compare_gt_outcomes


def _gt(class_id: int, box: list[float]) -> dict[str, object]:
    return {"class_id": class_id, "box": box}


def _pred(class_id: int, confidence: float, box: list[float]) -> dict[str, object]:
    return {"class_id": class_id, "confidence": confidence, "box": box}


def test_error_analysis_separates_low_confidence_localization_confusion_and_background_fp() -> None:
    ground_truth = [
        _gt(1, [0, 0, 10, 10]),
        _gt(0, [20, 0, 30, 10]),
        _gt(1, [40, 0, 50, 10]),
    ]
    predictions = [
        _pred(1, 0.20, [0, 0, 10, 10]),
        _pred(0, 0.70, [24, 0, 34, 10]),
        _pred(0, 0.80, [40, 0, 50, 10]),
        _pred(1, 0.90, [80, 80, 90, 90]),
    ]

    result = analyze_model_errors(
        ground_truth,
        predictions,
        image_size=(100, 100),
        confidence_threshold=0.25,
        matching_iou=0.5,
    )

    assert [item["reason"] for item in result["false_negatives"]] == [
        "low_confidence",
        "localization",
        "class_confusion",
    ]
    assert result["false_positives"] == [
        {
            "prediction_index": 1,
            "class_id": 0,
            "confidence": 0.7,
            "box": [24.0, 0.0, 34.0, 10.0],
            "reason": "near_object",
            "max_iou": 3 / 7,
            "nearest_gt_class_id": 0,
            "equivalent_size": 10.0,
            "edge": True,
        },
        {
            "prediction_index": 2,
            "class_id": 0,
            "confidence": 0.8,
            "box": [40.0, 0.0, 50.0, 10.0],
            "reason": "class_confusion",
            "max_iou": 1.0,
            "nearest_gt_class_id": 1,
            "equivalent_size": 10.0,
            "edge": True,
        },
        {
            "prediction_index": 3,
            "class_id": 1,
            "confidence": 0.9,
            "box": [80.0, 80.0, 90.0, 90.0],
            "reason": "background",
            "max_iou": 0.0,
            "nearest_gt_class_id": None,
            "equivalent_size": 10.0,
            "edge": False,
        },
    ]


def test_compare_gt_outcomes_reports_recovered_regressed_and_persistent_misses() -> None:
    ground_truth = [
        _gt(1, [0, 0, 8, 8]),
        _gt(1, [20, 0, 28, 8]),
        _gt(0, [40, 0, 48, 8]),
    ]
    e4 = [_pred(1, 0.9, [20, 0, 28, 8])]
    e6 = [_pred(1, 0.9, [0, 0, 8, 8])]

    transitions = compare_gt_outcomes(
        ground_truth,
        e4,
        e6,
        confidence_threshold=0.25,
        matching_iou=0.5,
    )

    assert [item["transition"] for item in transitions] == ["recovered", "regressed", "both_missed"]
    assert all(item["equivalent_size"] == 8.0 for item in transitions)
