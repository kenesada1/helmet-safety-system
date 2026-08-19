from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_train_script() -> object:
    spec = importlib.util.spec_from_file_location("train_baseline_e3", PROJECT_ROOT / "scripts" / "train_baseline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e3_training_protocol_names_model_capacity_as_the_only_core_variable() -> None:
    """E3 报告不能把固定的 imgsz=640 误写成实验变量。"""

    train_script = load_train_script()
    args = train_script.build_parser().parse_args(
        [
            "--model",
            str(PROJECT_ROOT / "artifacts" / "models" / "yolo11s.pt"),
            "--milestone",
            "M4.5",
            "--experiment-id",
            "E3",
            "--baseline-run",
            "baseline_yolo11n_001",
        ]
    )

    protocol = train_script.requested_training_protocol(args)

    assert protocol["core_variable"] == "model: YOLO11n -> YOLO11s"
    assert protocol["requested_batch"] == 8
    assert protocol["full_train_participates_each_epoch"] is True


def test_pretrained_model_metadata_records_real_bytes_and_sha256(tmp_path: Path) -> None:
    """训练证据必须能识别具体权重文件，而不只记录一个可替换的路径。"""

    train_script = load_train_script()
    model_path = tmp_path / "yolo11s.pt"
    model_path.write_bytes(b"official-weight-fixture")

    metadata = train_script.pretrained_model_metadata(model_path)

    assert metadata == {
        "path": str(model_path.resolve()),
        "bytes": 23,
        "sha256": hashlib.sha256(b"official-weight-fixture").hexdigest(),
    }


def test_e3_evaluator_cli_is_val_only_and_exposes_fixed_difficulty_rules() -> None:
    """E3 入口不得接受 test split，并须公开固定的 640/conf/IoU 与公平 batch=1 基准。"""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_m45_e3.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--training-report" in result.stdout
    assert "--split" not in result.stdout
    assert "imgsz=640" in result.stdout
    assert "conf=0.25" in result.stdout
    assert "IoU=0.5" in result.stdout
    assert "batch=1" in result.stdout


def test_dense_scene_analysis_counts_unmatched_predictions_as_false_positives() -> None:
    """密集场景报告不能只给 FN；同口径下未匹配预测必须计入 FP。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    ground_truth = [
        {"class_id": 0, "box": [index * 20.0, 0.0, index * 20.0 + 10.0, 10.0]}
        for index in range(10)
    ]
    predictions = [dict(item) for item in ground_truth[:8]] + [
        {"class_id": 1, "box": [500.0, 0.0, 510.0, 10.0]},
        {"class_id": 0, "box": [520.0, 0.0, 530.0, 10.0]},
    ]

    summary = e2.summarize_detection_slices(
        [{"image_id": "dense.jpg", "ground_truth": ground_truth, "predictions": predictions}],
        iou_threshold=0.5,
    )

    dense = summary["dense_scenes"]["ground_truth_gte_10"]
    assert dense["tp"] == 8
    assert dense["fn"] == 2
    assert dense["fp"] == 2


def test_slice_comparison_can_label_e3_without_changing_fixed_baseline_rules() -> None:
    """复用 E2 切片工具时，E3 数值不能被错误标成 E2/imgsz960。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    baseline = {
        "size_bins": {"10_lt_equivalent_size_le_20": {"recall": 0.4, "helmet_recall": 0.5, "no_helmet_recall": 0.3}},
        "dense_scenes": {"ground_truth_gte_10": {"recall": 0.7, "helmet_recall": 0.8, "no_helmet_recall": 0.6}},
    }
    candidate = {
        "size_bins": {"10_lt_equivalent_size_le_20": {"recall": 0.5, "helmet_recall": 0.55, "no_helmet_recall": 0.45}},
        "dense_scenes": {"ground_truth_gte_10": {"recall": 0.75, "helmet_recall": 0.81, "no_helmet_recall": 0.65}},
    }

    comparison = e2.compare_slice_recalls(baseline, candidate, candidate_key="m45_e3_640")

    item = comparison["size_bins"]["10_lt_equivalent_size_le_20"]["no_helmet_recall"]
    assert item == {
        "m4_baseline_640": 0.3,
        "m45_e3_640": 0.45,
        "absolute_change": 0.15,
        "percentage_point_change": 15.0,
    }
