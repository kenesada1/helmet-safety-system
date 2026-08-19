from __future__ import annotations

import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_train_script() -> object:
    spec = importlib.util.spec_from_file_location("train_baseline", PROJECT_ROOT / "scripts" / "train_baseline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_allocate_run_name_increments_numeric_suffix_without_overwriting(tmp_path: Path) -> None:
    """已有 _001 和 _002 目录时，新训练必须选择 _003。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    (tmp_path / "baseline_yolo11n_001").mkdir()
    (tmp_path / "baseline_yolo11n_002").mkdir()

    assert baseline.allocate_run_name(tmp_path, "baseline_yolo11n_001") == "baseline_yolo11n_003"


def test_training_entry_parses_custom_experiment_metadata() -> None:
    """M4.5 控制实验必须能显式记录实验身份，而不借用运行名推断。"""

    train_script = load_train_script()

    args = train_script.build_parser().parse_args(
        ["--milestone", "M4.5", "--experiment-id", "E1", "--baseline-run", "baseline_yolo11n_001"]
    )

    assert args.milestone == "M4.5"
    assert args.experiment_id == "E1"
    assert args.baseline_run == "baseline_yolo11n_001"


def test_training_report_metadata_uses_custom_values_and_keeps_m4_defaults() -> None:
    """报告不得把自定义 M4.5 元数据硬编码回 M4，旧命令仍应得到 M4 默认值。"""

    train_script = load_train_script()
    parser = train_script.build_parser()

    custom = train_script.experiment_metadata(
        parser.parse_args(
            ["--milestone", "M4.5", "--experiment-id", "E1", "--baseline-run", "baseline_yolo11n_001"]
        )
    )
    defaults = train_script.experiment_metadata(parser.parse_args([]))

    assert custom == {"milestone": "M4.5", "experiment_id": "E1", "baseline_run": "baseline_yolo11n_001"}
    assert defaults == {"milestone": "M4", "experiment_id": "", "baseline_run": ""}


def test_baseline_reference_snapshot_is_auditable_without_reading_it_as_training_input(tmp_path: Path) -> None:
    """指定 baseline_run 时，报告必须能证明该目录在实验前后没有变化。"""

    train_script = load_train_script()
    training_dir = tmp_path / "training"
    baseline_dir = training_dir / "baseline_yolo11n_001"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "results.csv").write_bytes(b"1234")
    (baseline_dir / "best.pt").write_bytes(b"123456")

    snapshot = train_script.baseline_reference_snapshot(training_dir, "baseline_yolo11n_001")

    assert snapshot["root"] == str(baseline_dir.resolve())
    assert snapshot["file_count"] == 2
    assert snapshot["total_bytes"] == 10
    assert snapshot["latest_mtime_ns"] is not None


def test_allocate_experiment_name_protects_both_run_directory_and_console_log(tmp_path: Path) -> None:
    """即使只有旧日志而没有训练目录，也不得复用同一个实验名。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    training_dir = tmp_path / "training"
    logs_dir = tmp_path / "logs"
    training_dir.mkdir()
    logs_dir.mkdir()
    (training_dir / "baseline_yolo11n_001").mkdir()
    (logs_dir / "baseline_yolo11n_002.log").write_text("old", encoding="utf-8")

    selected = baseline.allocate_experiment_name(training_dir, logs_dir, "baseline_yolo11n_001")

    assert selected == "baseline_yolo11n_003"
    assert sorted(path.name for path in training_dir.iterdir()) == ["baseline_yolo11n_001"]


def test_build_training_kwargs_preserves_the_reproducible_baseline_contract(tmp_path: Path) -> None:
    """基线入口不得意外改变图像尺寸、验证 split 或复现参数。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")

    kwargs = baseline.build_training_kwargs(
        data_yaml=tmp_path / "dataset.yaml",
        project_dir=tmp_path / "training",
        run_name="baseline_yolo11n_001",
        epochs=50,
        batch=8,
        device="0",
    )

    assert kwargs == {
        "data": str(tmp_path / "dataset.yaml"),
        "epochs": 50,
        "patience": 15,
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "device": "0",
        "seed": 42,
        "deterministic": True,
        "cache": False,
        "amp": True,
        "plots": True,
        "pretrained": True,
        "project": str(tmp_path / "training"),
        "name": "baseline_yolo11n_001",
        "exist_ok": False,
    }


def test_dataset_inventory_counts_images_labels_and_each_class(tmp_path: Path) -> None:
    """训练报告必须从实际标签统计图片、标签和两类框，而不是信任常量。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    processed = tmp_path / "processed"
    for split, classes in {"train": [0, 1, 1], "val": [0, 1], "test": [1]}.items():
        (processed / "images" / split).mkdir(parents=True)
        (processed / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (8, 8), "white").save(processed / "images" / split / "sample.jpg")
        (processed / "labels" / split / "sample.txt").write_text(
            "".join(f"{class_id} 0.5 0.5 0.2 0.2\n" for class_id in classes),
            encoding="utf-8",
        )

    inventory = baseline.dataset_inventory(processed)

    assert inventory["train"] == {
        "images": 1,
        "labels": 1,
        "boxes": 3,
        "class_boxes": {"helmet": 1, "no_helmet": 2},
    }
    assert inventory["val"]["class_boxes"] == {"helmet": 1, "no_helmet": 1}
    assert inventory["test"]["class_boxes"] == {"helmet": 0, "no_helmet": 1}


def test_analyze_training_results_uses_yolo_fitness_and_tolerates_short_fluctuation(tmp_path: Path) -> None:
    """最佳 epoch 应按 YOLO 检测 fitness 选择，末轮小波动不能误报过拟合。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),"
        "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
        "1,2.0,2.0,2.0,0.50,0.40,0.60,0.30,2.0,2.0,2.0\n"
        "2,1.5,1.5,1.5,0.55,0.45,0.65,0.35,1.8,1.8,1.8\n"
        "3,1.0,1.0,1.0,0.60,0.50,0.70,0.40,1.7,1.7,1.7\n"
        "4,0.8,0.8,0.8,0.59,0.49,0.68,0.38,1.9,1.9,1.9\n",
        encoding="utf-8",
    )

    summary = baseline.analyze_training_results(csv_path, requested_epochs=10)

    assert summary["best_epoch"] == 3
    assert summary["last_epoch"] == 4
    assert summary["early_stopping_triggered"] is True
    assert summary["stopping_reason"] == "early_stopping_before_requested_epochs"
    assert summary["best_val_metrics"] == {
        "precision": 0.6,
        "recall": 0.5,
        "map50": 0.7,
        "map50_95": 0.4,
    }
    assert summary["overfitting"]["detected"] is False
    assert summary["losses"]["train"]["box_loss"] == {"first": 2.0, "best": 1.0, "last": 0.8}
    assert summary["trailing_window"] == {
        "epoch_count": 4,
        "first_epoch": 1,
        "last_epoch": 4,
        "train_total_loss": {"first": 6.0, "last": 2.4, "change": -3.6, "slope_per_epoch": -1.2},
        "val_total_loss": {"first": 6.0, "last": 5.7, "change": -0.3, "slope_per_epoch": -0.1},
        "map50_95": {"first": 0.3, "last": 0.38, "change": 0.08, "slope_per_epoch": 0.026667},
        "precision": {"first": 0.5, "last": 0.59, "change": 0.09, "slope_per_epoch": 0.03},
        "recall": {"first": 0.4, "last": 0.49, "change": 0.09, "slope_per_epoch": 0.03},
    }


def test_analyze_training_results_rejects_non_finite_metrics(tmp_path: Path) -> None:
    """NaN 验证指标不能被写进一个成功的基线报告。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),"
        "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss\n"
        "1,1,1,1,0.5,0.4,nan,0.3,1,1,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite"):
        baseline.analyze_training_results(csv_path, requested_epochs=50)


def test_select_prediction_images_is_seeded_and_covers_both_classes(tmp_path: Path) -> None:
    """固定 seed 的可视化清单必须可复现，并尽可能覆盖两个类别。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    processed = tmp_path / "processed"
    (processed / "images" / "test").mkdir(parents=True)
    (processed / "labels" / "test").mkdir(parents=True)
    definitions = {"a": [0], "b": [1], "c": [0, 1], "d": [1], "e": [0], "f": []}
    for stem, classes in definitions.items():
        Image.new("RGB", (8, 8), "white").save(processed / "images" / "test" / f"{stem}.jpg")
        (processed / "labels" / "test" / f"{stem}.txt").write_text(
            "".join(f"{class_id} 0.5 0.5 0.2 0.2\n" for class_id in classes), encoding="utf-8"
        )

    selected = baseline.select_prediction_images(processed, count=4, seed=42)

    assert selected == baseline.select_prediction_images(processed, count=4, seed=42)
    selected_classes = {
        int(line.split()[0])
        for image_path in selected
        for line in (processed / "labels" / "test" / f"{image_path.stem}.txt").read_text().splitlines()
    }
    assert selected_classes == {0, 1}
    assert len(selected) == 4


def test_analyze_image_detections_separates_miss_false_positive_and_class_error() -> None:
    """同一张图里的漏检、误检和 helmet/no_helmet 混淆必须分别记录。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    ground_truth = [
        {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
        {"class_id": 1, "box": [20.0, 20.0, 30.0, 30.0]},
    ]
    predictions = [
        {"class_id": 1, "box": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9},
        {"class_id": 1, "box": [50.0, 50.0, 60.0, 60.0], "confidence": 0.8},
    ]

    result = baseline.analyze_image_detections(ground_truth, predictions, image_size=(100, 100))

    assert result["ground_truth_count"] == 2
    assert result["prediction_count"] == 2
    assert result["error_counts"]["class_confusion"] == 1
    assert result["error_counts"]["missed_detection"] == 1
    assert result["error_counts"]["false_positive"] == 1
    assert result["max_iou"] == pytest.approx(1.0)
    assert result["related_classes"] == ["helmet", "no_helmet"]
    assert result["missed_by_class"] == {"helmet": 0, "no_helmet": 1}
    assert result["false_positive_by_class"] == {"helmet": 0, "no_helmet": 1}
    assert result["class_confusion_pairs"] == {"helmet_as_no_helmet": 1}


def test_format_evaluation_metrics_keeps_overall_per_class_and_gpu_speed() -> None:
    """val/test 报告必须同时保留总体、逐类指标以及每图推理速度。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    summary = baseline.format_evaluation_metrics(
        {
            "metrics/precision(B)": 0.8,
            "metrics/recall(B)": 0.7,
            "metrics/mAP50(B)": 0.75,
            "metrics/mAP50-95(B)": 0.5,
            "fitness": 0.525,
        },
        [
            {"Class": "helmet", "Images": 10, "Instances": 12, "Box-P": 0.9, "Box-R": 0.8, "mAP50": 0.85, "mAP50-95": 0.6},
            {"Class": "no_helmet", "Images": 10, "Instances": 20, "Box-P": 0.7, "Box-R": 0.6, "mAP50": 0.65, "mAP50-95": 0.4},
        ],
        {"preprocess": 0.2, "inference": 4.0, "loss": 0.0, "postprocess": 0.8},
    )

    assert summary["overall"] == {"precision": 0.8, "recall": 0.7, "map50": 0.75, "map50_95": 0.5}
    assert summary["per_class"]["helmet"]["recall"] == 0.8
    assert summary["per_class"]["no_helmet"]["map50_95"] == 0.4
    assert summary["speed_ms_per_image"]["inference"] == 4.0
    assert summary["gpu_inference_images_per_second"] == 250.0


def test_write_json_report_refuses_to_overwrite_without_explicit_permission(tmp_path: Path) -> None:
    """同名报告存在时，默认必须失败，避免覆盖先前实验。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    report_path = tmp_path / "report.json"

    baseline.write_json_report(report_path, {"status": "running"})
    with pytest.raises(FileExistsError, match="overwrite"):
        baseline.write_json_report(report_path, {"status": "passed"})

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"status": "running"}


def test_validate_baseline_outputs_requires_all_requested_training_artifacts(tmp_path: Path) -> None:
    """缺少权重、结果曲线或混淆矩阵时，训练不能被标记为完整。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    run_dir = tmp_path / "baseline"
    (run_dir / "weights").mkdir(parents=True)
    for relative in (
        "weights/best.pt",
        "weights/last.pt",
        "results.csv",
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxF1_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
    ):
        (run_dir / relative).write_bytes(b"artifact")

    outputs = baseline.validate_baseline_outputs(run_dir)

    assert outputs["best_pt"] == str((run_dir / "weights" / "best.pt").resolve())
    (run_dir / "BoxR_curve.png").unlink()
    with pytest.raises(FileNotFoundError, match="BoxR_curve"):
        baseline.validate_baseline_outputs(run_dir)


@pytest.mark.parametrize(
    ("script_name", "required_options"),
    [
        ("train_baseline.py", ("--data", "--model", "--epochs", "--patience", "--batch", "--device")),
        ("evaluate_baseline.py", ("--training-report", "--split", "--prediction-count", "--conf")),
    ],
)
def test_m4_command_line_entries_expose_reproducibility_controls(
    script_name: str, required_options: tuple[str, ...]
) -> None:
    """M4 训练和评估入口应能直接运行，并公开关键复现参数。"""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert all(option in result.stdout for option in required_options)


def test_load_ground_truth_boxes_converts_normalized_yolo_coordinates(tmp_path: Path) -> None:
    """错误分析与真实框 JPG 必须把 YOLO 坐标准确还原到像素坐标。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    label_path = tmp_path / "sample.txt"
    label_path.write_text("1 0.5 0.25 0.2 0.1\n", encoding="utf-8")

    boxes = baseline.load_ground_truth_boxes(label_path, image_size=(100, 200))

    assert boxes == [{"class_id": 1, "box": [40.0, 40.0, 60.0, 60.0]}]


def test_save_ground_truth_visualization_creates_openable_jpeg(tmp_path: Path) -> None:
    """每个抽样测试图都应有可单独打开的真实框 JPG 用于人工对比。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    source = tmp_path / "source.jpg"
    output = tmp_path / "ground_truth" / "source.jpg"
    Image.new("RGB", (32, 24), "white").save(source)

    baseline.save_ground_truth_visualization(
        source,
        output,
        [{"class_id": 0, "box": [2.0, 3.0, 20.0, 18.0]}],
    )

    with Image.open(output) as rendered:
        rendered.verify()
    assert output.stat().st_size > 0


def test_scan_training_log_flags_jpeg_repair_cache_version_and_corruption(tmp_path: Path) -> None:
    """关键数据告警必须进入结构化报告，不能藏在长控制台日志中。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "WARNING corrupt JPEG restored and saved\nWARNING cache version mismatch\n2 corrupt images\n",
        encoding="utf-8",
    )

    audit = baseline.scan_training_log(log_path)

    assert audit["jpeg_auto_repair_warning"] is True
    assert audit["cache_version_warning"] is True
    assert audit["corrupt_data_warning"] is True


def test_scan_training_log_does_not_treat_zero_corrupt_progress_as_warning(tmp_path: Path) -> None:
    baseline = importlib.import_module("helmet_safety.training.baseline")
    log_path = tmp_path / "clean.log"
    log_path.write_text(
        "train: Scanning labels/train... 5457 images, 0 backgrounds, 0 corrupt: 100%\n"
        "val: Scanning labels/val... 607 images, 0 backgrounds, 0 corrupt: 100%\n",
        encoding="utf-8",
    )

    audit = baseline.scan_training_log(log_path)

    assert audit["corrupt_data_warning"] is False
    assert audit["warning_lines"] == []
