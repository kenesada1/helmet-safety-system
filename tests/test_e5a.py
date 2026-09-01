from __future__ import annotations

from collections import Counter
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def test_tiny_difficulty_matches_against_every_ground_truth_box() -> None:
    """若只加载微小真实框导致大框预测被错配，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    records = [
        {
            "image_id": "train-a.jpg",
            "image_path": "/train/train-a.jpg",
            "ground_truth": [
                {"class_id": 0, "box": [0.0, 0.0, 40.0, 40.0]},
                {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [50.0, 50.0, 56.0, 56.0]},
            ],
            "predictions": [
                {"class_id": 0, "box": [0.0, 0.0, 40.0, 40.0]},
                {"class_id": 1, "box": [50.0, 50.0, 56.0, 56.0]},
            ],
        }
    ]

    rows = e5a.identify_tiny_difficult_samples(records, max_equivalent_size=10.0, iou_threshold=0.5)

    assert rows == [
        {
            "image_id": "train-a.jpg",
            "image_path": "/train/train-a.jpg",
            "tiny_ground_truth": 2,
            "tiny_helmet_ground_truth": 1,
            "tiny_no_helmet_ground_truth": 1,
            "tiny_missed": 1,
            "tiny_helmet_missed": 1,
            "tiny_no_helmet_missed": 0,
            "sampling_weight": 2,
        }
    ]


def test_resampled_manifests_are_reproducible_and_keep_every_training_image() -> None:
    """若清单长度、全量基线或固定种子可复现性被破坏，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    images = [Path("h.jpg"), Path("n.jpg"), Path("x.jpg")]
    difficult = [
        {"image_path": "h.jpg", "sampling_weight": 2},
        {"image_path": "n.jpg", "sampling_weight": 1},
    ]

    first = e5a.build_resampled_manifests(images, difficult, extra_count=8, seed=42)
    second = e5a.build_resampled_manifests(images, difficult, extra_count=8, seed=42)

    assert first == second
    assert first["control_extra"] == ["n.jpg", "h.jpg", "h.jpg", "h.jpg", "x.jpg", "x.jpg", "x.jpg", "h.jpg"]
    assert first["e5a_extra"] == ["h.jpg", "h.jpg", "h.jpg", "h.jpg", "n.jpg", "n.jpg", "n.jpg", "h.jpg"]
    assert len(first["control_entries"]) == 11
    assert len(first["e5a_entries"]) == 11
    for image in ("h.jpg", "n.jpg", "x.jpg"):
        assert Counter(first["control_entries"])[image] >= 1
        assert Counter(first["e5a_entries"])[image] >= 1


def test_manifest_audit_rejects_validation_or_test_images(tmp_path: Path) -> None:
    """若训练清单混入验证集或测试集，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    root = tmp_path / "processed"
    train = root / "images" / "train"
    val = root / "images" / "val"
    labels = root / "labels" / "train"
    train.mkdir(parents=True)
    val.mkdir(parents=True)
    labels.mkdir(parents=True)
    train_image = train / "a.jpg"
    val_image = val / "v.jpg"
    train_image.write_bytes(b"image")
    val_image.write_bytes(b"image")
    (labels / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    good = e5a.audit_training_manifest(
        [str(train_image), str(train_image)],
        original_train_images=[train_image],
        processed_root=root,
        expected_entries=2,
    )
    assert good["entries"] == 2
    assert good["unique_images"] == 1
    assert good["all_original_training_images_present"] is True
    assert good["validation_images_present"] is False
    assert good["test_images_present"] is False

    with pytest.raises(ValueError, match="验证集或测试集"):
        e5a.audit_training_manifest(
            [str(train_image), str(val_image)],
            original_train_images=[train_image],
            processed_root=root,
            expected_entries=2,
        )


def test_training_kwargs_lock_independent_start_and_requested_hyperparameters(tmp_path: Path) -> None:
    """若训练恢复E4、覆盖E4或偏离指定参数，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    e4_args = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "close_mosaic": 10,
    }

    kwargs = e5a.build_e5a_training_kwargs(
        e4_args=e4_args,
        data_yaml=tmp_path / "dataset.yaml",
        project_dir=tmp_path / "training",
        run_name="e5a_control",
        device="0",
    )

    assert kwargs["resume"] is False
    assert kwargs["exist_ok"] is False
    assert kwargs["epochs"] == 30
    assert kwargs["patience"] == 10
    assert kwargs["imgsz"] == 960
    assert kwargs["batch"] == 2
    assert kwargs["workers"] == 0
    assert kwargs["optimizer"] == "AdamW"
    assert kwargs["lr0"] == 0.0005
    assert kwargs["lrf"] == 0.01
    assert kwargs["momentum"] == 0.9
    assert kwargs["weight_decay"] == 0.0005
    assert kwargs["warmup_epochs"] == 1
    assert kwargs["seed"] == 42
    assert kwargs["deterministic"] is True
    assert kwargs["amp"] is True
    assert {key: kwargs[key] for key in e5a.E4_AUGMENTATION_KEYS} == e4_args


def test_chinese_evaluation_report_uses_full_metric_names() -> None:
    """若用户报告重新出现检测计数英文缩写，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    metrics = {
        "overall": {"tp": 8, "fn": 2, "fp": 1, "precision": 8 / 9, "recall": 0.8, "f1": 16 / 19},
        "per_class": {
            "helmet": {"tp": 3, "fn": 1, "fp": 1, "precision": 0.75, "recall": 0.75, "f1": 0.75},
            "no_helmet": {"tp": 5, "fn": 1, "fp": 0, "precision": 1.0, "recall": 5 / 6, "f1": 10 / 11},
        },
        "tiny": {"tp": 1, "fn": 1, "fp": 0, "precision": 1.0, "recall": 0.5, "f1": 2 / 3},
    }

    report = e5a.render_chinese_evaluation_markdown({"E4": metrics, "延长训练对照": metrics, "E5-A": metrics})

    assert "正确检出" in report
    assert "漏检" in report
    assert "误检" in report
    assert "查准率" in report
    assert "召回率" in report
    assert "综合指标" in report
    for forbidden in (" TP ", " FN ", " FP ", "| TP", "| FN", "| FP"):
        assert forbidden not in report


def test_full_evaluation_counts_only_unmatched_tiny_predictions_as_tiny_false_positives() -> None:
    """若微小目标误检错误计入大框预测或忽略未匹配小框，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    records = [
        {
            "image_id": "v.jpg",
            "ground_truth": [
                {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [20.0, 20.0, 60.0, 60.0]},
            ],
            "predictions": [
                {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [20.0, 20.0, 60.0, 60.0]},
                {"class_id": 0, "box": [70.0, 70.0, 76.0, 76.0]},
                {"class_id": 1, "box": [80.0, 80.0, 120.0, 120.0]},
            ],
        }
    ]

    metrics = e5a.summarize_e5a_evaluation(records, matching_iou=0.5, max_equivalent_size=10.0)

    assert metrics["overall"]["tp"] == 2
    assert metrics["overall"]["fp"] == 2
    assert metrics["tiny"] == {
        "tp": 1,
        "fn": 0,
        "fp": 1,
        "precision": 0.5,
        "recall": 1.0,
        "f1": 2 / 3,
    }


def test_user_report_payload_uses_chinese_detection_count_keys() -> None:
    """若机器可读用户报告仍暴露正确检出、漏检、误检的英文缩写，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    payload = {
        "metrics": {
            "overall": {"tp": 10, "fn": 2, "fp": 3, "precision": 10 / 13},
            "tiny": {"tp": 1, "fn": 1, "fp": 2},
        }
    }

    translated = e5a.chinese_detection_count_keys(payload)

    assert translated["metrics"]["overall"] == {
        "正确检出": 10,
        "漏检": 2,
        "误检": 3,
        "precision": 10 / 13,
    }
    assert translated["metrics"]["tiny"] == {"正确检出": 1, "漏检": 1, "误检": 2}
    serialized = json.dumps(translated, ensure_ascii=False)
    for forbidden in ('"tp"', '"fn"', '"fp"'):
        assert forbidden not in serialized


def test_e4_anchor_gate_requires_candidate_c_full_validation_counts() -> None:
    """若E4候选C完整验证锚点漂移后仍允许正式训练，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    valid = {"overall": {"tp": 9420, "fn": 505, "fp": 951}, "tiny": {"tp": 79, "fn": 49}}

    assert e5a.validate_e4_candidate_c_anchor(valid) == {
        "overall_correct_detections": 9420,
        "overall_misses": 505,
        "overall_false_detections": 951,
        "tiny_correct_detections": 79,
        "tiny_misses": 49,
    }
    with pytest.raises(RuntimeError, match="E4候选C验证锚点"):
        e5a.validate_e4_candidate_c_anchor(
            {"overall": {"tp": 9419, "fn": 506, "fp": 951}, "tiny": {"tp": 79, "fn": 49}}
        )


def test_effectiveness_requires_tiny_recall_and_f1_gain_without_material_overall_loss() -> None:
    """若只靠单项改善就宣称重采样有效，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    control = {"overall": {"f1": 0.930}, "tiny": {"recall": 0.60, "f1": 0.50}}

    effective = e5a.judge_resampling_effectiveness(
        control,
        {"overall": {"f1": 0.929}, "tiny": {"recall": 0.62, "f1": 0.53}},
    )
    assert effective["effective"] is True
    assert effective["overall_f1_material_loss"] is False

    ineffective = e5a.judge_resampling_effectiveness(
        control,
        {"overall": {"f1": 0.925}, "tiny": {"recall": 0.62, "f1": 0.53}},
    )
    assert ineffective["effective"] is False
    assert ineffective["overall_f1_material_loss"] is True


def test_e5a_cli_exposes_only_gated_experiment_phases() -> None:
    """若入口允许测试集、恢复训练或绕过阶段门禁，本测试必须失败。"""

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "train" / "pipeline_e5a.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for phase in ("prepare", "smoke", "anchor", "train-control", "train-e5a", "evaluate"):
        assert phase in result.stdout
    assert "--split" not in result.stdout
    assert "--resume" not in result.stdout
    assert "test" not in result.stdout.lower()


def test_staged_dataset_isolated_from_original_caches_and_omits_test_split(tmp_path: Path) -> None:
    """若训练缓存可能写回原数据或派生配置暴露测试集，本测试必须失败。"""

    e5a = importlib.import_module("helmet_safety.training.e5a")
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    train_image = source / "images" / "train" / "a.jpg"
    val_image = source / "images" / "val" / "v.jpg"
    train_image.write_bytes(b"train-image")
    val_image.write_bytes(b"val-image")
    source_label = source / "labels" / "train" / "a.txt"
    source_label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (source / "labels" / "val" / "v.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    stage = tmp_path / "stage"

    result = e5a.stage_dataset_variant(
        source_root=source,
        stage_root=stage,
        manifest_entries=[str(train_image), str(train_image)],
        expected_train_images=1,
        expected_val_images=1,
        expected_manifest_entries=2,
    )

    assert Path(result["dataset_yaml"]).is_file()
    yaml_text = Path(result["dataset_yaml"]).read_text(encoding="utf-8")
    assert "train:" in yaml_text and "val:" in yaml_text
    assert "test:" not in yaml_text
    assert (stage / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == source_label.read_text(encoding="utf-8")
    assert not (source / "labels" / "train.cache").exists()
    assert Path(result["train_manifest"]).read_text(encoding="utf-8").splitlines() == [
        str((stage / "images" / "train" / "a.jpg").resolve()),
        str((stage / "images" / "train" / "a.jpg").resolve()),
    ]
