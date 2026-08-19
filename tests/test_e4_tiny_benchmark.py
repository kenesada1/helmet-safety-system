from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from helmet_safety.training import m45_e2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, size: tuple[int, int]) -> None:
    Image.new("RGB", size, "white").save(path)


def test_select_tiny_val_images_uses_original_pixel_size_and_inclusive_boundary(tmp_path: Path) -> None:
    images_dir = tmp_path / "images" / "val"
    labels_dir = tmp_path / "labels" / "val"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    _write_image(images_dir / "boundary.jpg", (100, 50))
    _write_image(images_dir / "large.jpg", (100, 50))
    (labels_dir / "boundary.txt").write_text("1 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    (labels_dir / "large.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")

    selected = m45_e2.select_tiny_val_images(images_dir, labels_dir, max_equivalent_size=10.0)

    assert selected == [
        {
            "image_id": "boundary.jpg",
            "image_path": str((images_dir / "boundary.jpg").resolve()),
            "image_width": 100,
            "image_height": 50,
            "tiny_gt_indices": [0],
            "tiny_gt_count": 1,
        }
    ]


def test_tiny_summary_matches_all_ground_truth_before_counting_tiny_only() -> None:
    records = [
        {
            "image_id": "overlap.jpg",
            "ground_truth": [
                {"class_id": 1, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [0.0, 0.0, 14.0, 14.0]},
            ],
            "predictions": [{"class_id": 1, "box": [0.0, 0.0, 14.0, 14.0]}],
        }
    ]

    result = m45_e2.summarize_tiny_ground_truth(records, max_equivalent_size=10.0, iou_threshold=0.5)

    assert result["summary"] == {
        "images": 1,
        "ground_truth_instances": 1,
        "helmet_instances": 0,
        "no_helmet_instances": 1,
        "tp": 0,
        "fn": 1,
        "recall": 0.0,
        "helmet_tp": 0,
        "helmet_fn": 0,
        "helmet_recall": None,
        "no_helmet_tp": 0,
        "no_helmet_fn": 1,
        "no_helmet_recall": 0.0,
    }
    assert result["false_negatives"] == [
        {
            "image_id": "overlap.jpg",
            "gt_index": 0,
            "class_id": 1,
            "class_name": "no_helmet",
            "gt_box_xyxy": [0.0, 0.0, 10.0, 10.0],
        }
    ]


def test_assert_expected_tiny_summary_stops_on_mismatch() -> None:
    actual = {
        "ground_truth_instances": 128,
        "tp": 76,
        "fn": 52,
        "helmet_instances": 14,
        "helmet_tp": 4,
        "helmet_fn": 10,
        "no_helmet_instances": 114,
        "no_helmet_tp": 72,
        "no_helmet_fn": 42,
    }

    with pytest.raises(RuntimeError, match="tiny benchmark mismatch"):
        m45_e2.assert_expected_tiny_summary(actual)


def test_e4_tiny_cli_exposes_only_the_fixed_val_protocol() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "reproduce_e4_tiny.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "val only" in result.stdout
    assert "imgsz=960" in result.stdout
    assert "conf=0.25" in result.stdout
    assert "matching IoU=0.5" in result.stdout
    for forbidden in ("--split", "--imgsz", "--conf", "--matching-iou", "--weights"):
        assert forbidden not in result.stdout


def test_conf_evaluation_uses_all_gt_for_fp_and_identifies_recovered_baseline_fn() -> None:
    records = [
        {
            "image_id": "sample.jpg",
            "ground_truth": [
                {"class_id": 1, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 0, "box": [20.0, 20.0, 40.0, 40.0]},
            ],
            "predictions": [
                {"class_id": 1, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 0, "box": [20.0, 20.0, 40.0, 40.0]},
                {"class_id": 1, "box": [60.0, 60.0, 70.0, 70.0]},
            ],
        }
    ]
    baseline_fn = [
        {
            "image_id": "sample.jpg",
            "gt_index": 0,
            "class_id": 1,
            "class_name": "no_helmet",
            "gt_box_xyxy": [0.0, 0.0, 10.0, 10.0],
        }
    ]

    result = m45_e2.evaluate_tiny_conf_records(records, baseline_fn, max_equivalent_size=10.0, iou_threshold=0.5)

    assert result["summary"]["tp"] == 1
    assert result["summary"]["fn"] == 0
    assert result["false_positives"] == 1
    assert result["recovered_original_fn_count"] == 1
    assert result["recovered_helmet_count"] == 0
    assert result["recovered_no_helmet_count"] == 1
    assert result["recovered_original_false_negatives"] == baseline_fn


def test_conf_sweep_markdown_contains_comparison_and_recovered_ids() -> None:
    rows = [
        {
            "conf": 0.2,
            "tiny_tp": 80,
            "tiny_fn": 48,
            "tiny_recall": 0.625,
            "helmet_tp": 5,
            "helmet_fn": 9,
            "helmet_recall": 5 / 14,
            "no_helmet_tp": 75,
            "no_helmet_fn": 39,
            "no_helmet_recall": 75 / 114,
            "fp_35_images": 100,
            "fp_delta_vs_025": 20,
            "fp_ratio_vs_025": 1.25,
            "recovered_original_fn_count": 3,
            "recovered_helmet_count": 1,
            "recovered_no_helmet_count": 2,
            "recovered_original_fn_keys": "a.jpg:0; b.jpg:1; c.jpg:2",
        }
    ]

    markdown = m45_e2.render_tiny_conf_markdown(rows, analysis="结论文本")

    assert "| 0.2 | 80 | 48 | 0.625000" in markdown
    assert "a.jpg:0; b.jpg:1; c.jpg:2" in markdown
    assert "结论文本" in markdown


def test_e4_tiny_conf_sweep_cli_locks_all_seven_conf_values() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "sweep_e4_tiny_conf.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for value in ("0.25", "0.20", "0.15", "0.10", "0.05", "0.01", "0.001"):
        assert value in result.stdout
    for forbidden in ("--split", "--imgsz", "--conf", "--matching-iou", "--weights"):
        assert forbidden not in result.stdout


def test_duplicate_box_audit_counts_only_same_class_pairs_at_or_above_threshold() -> None:
    records = [
        {
            "image_id": "dense.jpg",
            "predictions": [
                {"class_id": 1, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [20.0, 20.0, 30.0, 30.0]},
            ],
        }
    ]

    result = m45_e2.audit_obvious_duplicate_boxes(records, duplicate_iou_threshold=0.7)

    assert result["duplicate_pairs"] == 1
    assert result["images_with_duplicates"] == 1
    assert result["predictions_in_duplicate_pairs"] == 2
    assert result["has_obvious_duplicates"] is True
    assert result["details"] == [
        {
            "image_id": "dense.jpg",
            "class_id": 1,
            "class_name": "no_helmet",
            "prediction_index_a": 0,
            "prediction_index_b": 1,
            "pair_iou": 1.0,
        }
    ]


def test_nms_iou_markdown_contains_duplicate_audit_and_recovered_ids() -> None:
    rows = [
        {
            "nms_iou": 0.8,
            "tiny_tp": 79,
            "tiny_fn": 49,
            "tiny_recall": 79 / 128,
            "helmet_recall": 5 / 14,
            "no_helmet_recall": 74 / 114,
            "fp_35_images": 280,
            "recovered_original_fn_count": 2,
            "recovered_helmet_count": 1,
            "recovered_no_helmet_count": 1,
            "obvious_duplicate_pairs": 4,
            "images_with_obvious_duplicates": 2,
            "has_obvious_duplicates": True,
            "recovered_original_fn_keys": "a.jpg:0:helmet; b.jpg:1:no_helmet",
        }
    ]

    markdown = m45_e2.render_tiny_nms_iou_markdown(rows, analysis="NMS结论")

    assert "| 0.80 | 79 | 49 | 0.617188" in markdown
    assert "| 4 | 2 | 是 |" in markdown
    assert "a.jpg:0:helmet; b.jpg:1:no_helmet" in markdown
    assert "NMS结论" in markdown


def test_e4_tiny_nms_iou_sweep_cli_locks_all_six_values() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "sweep_e4_tiny_nms_iou.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for value in ("0.50", "0.60", "0.70", "0.80", "0.90", "0.95"):
        assert value in result.stdout
    for forbidden in ("--split", "--imgsz", "--conf", "--matching-iou", "--nms-iou", "--weights"):
        assert forbidden not in result.stdout


@pytest.mark.parametrize(
    ("max_iou", "expected"),
    [
        (0.5, "iou_ge_0_5"),
        (0.4999, "iou_0_4_to_0_5"),
        (0.4, "iou_0_4_to_0_5"),
        (0.3999, "iou_0_2_to_0_4"),
        (0.2, "iou_0_2_to_0_4"),
        (0.1999, "iou_0_1_to_0_2"),
        (0.1, "iou_0_1_to_0_2"),
        (0.0999, "unresolved"),
        (None, "unresolved"),
    ],
)
def test_localization_bucket_boundaries(max_iou: float | None, expected: str) -> None:
    assert m45_e2.localization_bucket_for_iou(max_iou) == expected


def test_fn_candidate_analysis_keeps_confidence_and_iou_maxima_independent() -> None:
    original_false_negatives = [
        {
            "image_id": "a.jpg",
            "gt_index": 0,
            "class_id": 1,
            "class_name": "no_helmet",
            "gt_box_xyxy": [0.0, 0.0, 10.0, 10.0],
        },
        {
            "image_id": "b.jpg",
            "gt_index": 2,
            "class_id": 0,
            "class_name": "helmet",
            "gt_box_xyxy": [0.0, 0.0, 10.0, 10.0],
        },
    ]
    predictions_by_image = {
        "a.jpg": [
            {"class_id": 1, "confidence": 0.9, "box": [20.0, 20.0, 30.0, 30.0]},
            {"class_id": 1, "confidence": 0.05, "box": [0.0, 0.0, 8.0, 10.0]},
            {"class_id": 0, "confidence": 0.8, "box": [20.0, 20.0, 30.0, 30.0]},
            {"class_id": 0, "confidence": 0.02, "box": [0.0, 0.0, 10.0, 10.0]},
        ],
        "b.jpg": [
            {"class_id": 1, "confidence": 0.3, "box": [20.0, 20.0, 30.0, 30.0]},
        ],
    }

    result = m45_e2.analyze_fn_candidate_evidence(original_false_negatives, predictions_by_image)

    first = result["details"][0]
    assert first["correct_class_candidate_count"] == 2
    assert first["correct_class_max_confidence"] == 0.9
    assert first["correct_class_max_confidence_candidate_iou"] == 0.0
    assert first["correct_class_max_iou"] == 0.8
    assert first["correct_class_max_iou_candidate_confidence"] == 0.05
    assert first["wrong_class_candidate_count"] == 2
    assert first["wrong_class_max_confidence"] == 0.8
    assert first["wrong_class_max_confidence_candidate_iou"] == 0.0
    assert first["wrong_class_max_iou"] == 1.0
    assert first["wrong_class_max_iou_candidate_confidence"] == 0.02
    assert first["localization_bucket"] == "iou_ge_0_5"
    second = result["details"][1]
    assert second["correct_class_max_confidence"] is None
    assert second["correct_class_max_iou"] is None
    assert second["localization_bucket"] == "unresolved"
    assert result["summary"]["overall"]["total"] == 2
    assert result["summary"]["overall"]["buckets"]["iou_ge_0_5"] == {"count": 1, "share": 0.5}
    assert result["summary"]["helmet"]["buckets"]["unresolved"] == {"count": 1, "share": 1.0}
    assert result["summary"]["no_helmet"]["buckets"]["iou_ge_0_5"] == {"count": 1, "share": 1.0}


def test_e4_fn_localization_cli_locks_low_conf_and_default_nms() -> None:
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "analyze_e4_fn_localization.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "conf=0.001" in result.stdout
    assert "matching IoU=0.5" in result.stdout
    assert "default NMS" in result.stdout
    for forbidden in ("--split", "--imgsz", "--conf", "--matching-iou", "--nms-iou", "--weights"):
        assert forbidden not in result.stdout
