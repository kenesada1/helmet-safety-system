from __future__ import annotations

from collections import Counter
import importlib
from pathlib import Path
import subprocess
import sys

import pytest


def _e5b():
    return importlib.import_module("helmet_safety.training.e5b")


def test_context_window_hits_requested_scale_and_keeps_center_box_complete() -> None:
    """若窗口尺度计算错误或中心框在图边缘被裁断，本测试必须失败。"""

    e5b = _e5b()
    centered = e5b.compute_context_window(
        image_size=(1000, 800), center_box=(495.0, 395.0, 505.0, 405.0), target_size=16.0
    )
    edge = e5b.compute_context_window(
        image_size=(1000, 800), center_box=(0.0, 0.0, 4.0, 4.0), target_size=16.0
    )

    assert centered == (200, 100, 800, 700)
    assert e5b.equivalent_size_after_resize((495.0, 395.0, 505.0, 405.0), centered) == 16.0
    assert edge == (0, 0, 240, 240)
    assert edge[0] <= 0.0 and edge[1] <= 0.0 and edge[2] >= 4.0 and edge[3] >= 4.0


def test_context_window_preserves_minimum_scene_context_without_over_enlarging() -> None:
    """若极小框被裁得过紧、在960输入下放大超过24像素，本测试必须失败。"""

    e5b = _e5b()
    window = e5b.compute_context_window(
        image_size=(500, 400), center_box=(249.0, 199.0, 251.0, 201.0), target_size=18.0
    )

    assert window == (186, 136, 314, 264)
    assert window[2] - window[0] == 128
    assert e5b.equivalent_size_after_resize((249.0, 199.0, 251.0, 201.0), window) == 15.0
    assert e5b.equivalent_size_after_resize((249.0, 199.0, 251.0, 201.0), window) <= 24.0


def test_crop_label_conversion_keeps_only_qualified_intersections() -> None:
    """若中心框不完整、低可见率框未删除或合格相交框未转换，本测试必须失败。"""

    e5b = _e5b()
    boxes = [
        {"class_id": 0, "box": [40.0, 40.0, 60.0, 60.0]},
        {"class_id": 1, "box": [100.0, 100.0, 125.0, 125.0]},
        {"class_id": 1, "box": [100.0, 100.0, 130.0, 130.0]},
        {"class_id": 1, "box": [119.0, 40.0, 121.0, 42.0]},
    ]

    converted, decisions = e5b.convert_boxes_for_crop(
        boxes, center_index=0, crop_window=(20, 20, 120, 120)
    )

    assert converted == [
        {"source_index": 0, "class_id": 0, "box": [20.0, 20.0, 40.0, 40.0], "is_center": True},
        {"source_index": 1, "class_id": 1, "box": [80.0, 80.0, 100.0, 100.0], "is_center": False},
    ]
    assert decisions == [
        {"source_index": 0, "action": "保留", "visible_ratio": 1.0, "clipped": False},
        {"source_index": 1, "action": "保留", "visible_ratio": 0.64, "clipped": True},
        {"source_index": 2, "action": "删除", "visible_ratio": pytest.approx(4 / 9), "clipped": True},
        {"source_index": 3, "action": "删除", "visible_ratio": 0.5, "clipped": True},
    ]


def test_crop_selection_uses_all_images_first_and_prioritizes_tiny_helmets() -> None:
    """若选择仅来自历史漏检、突破单图上限或漏掉可覆盖安全帽，本测试必须失败。"""

    e5b = _e5b()
    records = [
        {
            "image_path": "a.jpg",
            "width": 1200,
            "height": 900,
            "boxes": [
                {"class_id": 0, "box": [100.0, 100.0, 108.0, 108.0]},
                {"class_id": 1, "box": [200.0, 100.0, 208.0, 108.0]},
                {"class_id": 0, "box": [300.0, 100.0, 308.0, 108.0]},
                {"class_id": 1, "box": [400.0, 100.0, 408.0, 108.0]},
                {"class_id": 0, "box": [500.0, 100.0, 508.0, 108.0]},
            ],
        },
        {
            "image_path": "b.jpg",
            "width": 1200,
            "height": 900,
            "boxes": [{"class_id": 1, "box": [600.0, 300.0, 608.0, 308.0]}],
        },
    ]

    requests = e5b.select_context_crop_requests(records)

    assert Counter(row["image_path"] for row in requests) == {"a.jpg": 4, "b.jpg": 2}
    assert [row["center_index"] for row in requests if row["image_path"] == "a.jpg"] == [0, 2, 4, 1]
    assert [row["center_index"] for row in requests if row["image_path"] == "b.jpg"] == [0, 0]
    assert Counter((row["image_path"], row["center_index"]) for row in requests).most_common(1)[0][1] <= 2
    assert {row["target_size"] for row in requests if row["image_path"] == "b.jpg"} == {14.0, 16.0}
    assert max(row["center_size_960"] for row in requests) <= 16.0


def test_crop_selection_can_shift_primary_centers_to_16_without_raising_hard_maximum() -> None:
    """若尺度校准仍生成14像素中心或把基础中心推过16像素，本测试必须失败。"""

    e5b = _e5b()
    records = [
        {
            "image_path": "a.jpg",
            "width": 1200,
            "height": 900,
            "boxes": [
                {"class_id": 0, "box": [100.0, 100.0, 108.0, 108.0]},
                {"class_id": 1, "box": [200.0, 100.0, 208.0, 108.0]},
            ],
        }
    ]

    requests = e5b.select_context_crop_requests(
        records, primary_target_size=16.0, repeated_target_size=16.0
    )

    assert len(requests) == 4
    assert {row["target_size"] for row in requests} == {16.0}
    assert max(row["center_size_960"] for row in requests) <= 16.0


def test_manifest_uses_uniform_control_fallback_without_repeating_crop_entries() -> None:
    """若合格裁剪不足时放宽集中度或改用困难整图补齐，本测试必须失败。"""

    e5b = _e5b()
    manifest = e5b.build_context_crop_manifest(
        [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")],
        [Path("crop-1.jpg"), Path("crop-2.jpg")],
        extra_count=4,
        seed=42,
    )

    assert manifest["crop_entries"] == ["crop-1.jpg", "crop-2.jpg"]
    assert manifest["uniform_fallback"] == ["b.jpg", "a.jpg"]
    assert manifest["entries"] == ["a.jpg", "b.jpg", "c.jpg", "crop-1.jpg", "crop-2.jpg", "b.jpg", "a.jpg"]


def test_crop_mapping_audit_rejects_usage_caps_and_oversized_centers() -> None:
    """若每图/每目标集中度或24像素硬上限失守后仍通过门禁，本测试必须失败。"""

    e5b = _e5b()
    valid = [
        {"image_path": "a.jpg", "center_index": 0, "center_size_960": 16.0},
        {"image_path": "a.jpg", "center_index": 0, "center_size_960": 18.0},
        {"image_path": "a.jpg", "center_index": 1, "center_size_960": 20.0},
        {"image_path": "a.jpg", "center_index": 2, "center_size_960": 23.0},
    ]
    audit = e5b.audit_crop_mappings(valid)
    assert audit["maximum_crops_per_image"] == 4
    assert audit["maximum_uses_per_target"] == 2

    with pytest.raises(ValueError, match="单张原图"):
        e5b.audit_crop_mappings([*valid, {"image_path": "a.jpg", "center_index": 3, "center_size_960": 16.0}])
    with pytest.raises(ValueError, match="单个真实目标"):
        e5b.audit_crop_mappings(valid[:2] + [{"image_path": "a.jpg", "center_index": 0, "center_size_960": 19.0}])
    with pytest.raises(ValueError, match="24"):
        e5b.audit_crop_mappings([{"image_path": "a.jpg", "center_index": 0, "center_size_960": 24.01}])


def test_augmentation_gate_requires_1000_tracked_centers() -> None:
    """若只审计很多增强样本、却没有追踪够1000个中心框便放行，本测试必须失败。"""

    e5b = _e5b()

    insufficient = e5b.evaluate_augmented_center_gate([14.0] * 999)
    sufficient = e5b.evaluate_augmented_center_gate([14.0] * 1000)

    assert insufficient == {
        "minimum_tracked_centers": 1000,
        "tracked_center_boxes": 999,
        "minimum_in_required_range_ratio": 0.4,
        "in_required_range_ratio": 1.0,
        "median_in_required_range": True,
        "p90_within_required_range": True,
        "hard_maximum_respected": True,
        "passed": False,
    }
    assert sufficient["passed"] is True


def test_augmentation_gate_accepts_control_compatible_central_envelope() -> None:
    """若固定随机缩放下中位数、九成分位和硬上限均合格却仍被旧50%交集门槛拦截，本测试必须失败。"""

    e5b = _e5b()
    acceptable = [10.0] * 464 + [14.0] * 436 + [22.0] * 100
    excessive_upper_tail = [10.0] * 464 + [14.0] * 400 + [22.0] * 136

    accepted = e5b.evaluate_augmented_center_gate(acceptable)
    rejected = e5b.evaluate_augmented_center_gate(excessive_upper_tail)

    assert accepted["in_required_range_ratio"] == 0.436
    assert accepted["p90_within_required_range"] is True
    assert accepted["passed"] is True
    assert rejected["p90_within_required_range"] is False
    assert rejected["passed"] is False


def test_augmentation_audit_stops_at_center_target_or_safety_limit() -> None:
    """若增强审计在中心框不足时过早停止，或超过安全样本上限仍继续，本测试必须失败。"""

    e5b = _e5b()

    assert e5b.should_continue_augmentation_audit(999, 11_999) is True
    assert e5b.should_continue_augmentation_audit(1_000, 7_321) is False
    assert e5b.should_continue_augmentation_audit(999, 12_000) is False


def test_experiment_root_requires_a_new_safe_e5b_run_name(tmp_path: Path) -> None:
    """若新审计能覆盖001记录或通过路径穿越写出实验目录，本测试必须失败。"""

    e5b = _e5b()

    assert e5b.resolve_e5b_experiment_root(tmp_path, "e5b_context_crop_002") == (
        tmp_path / "artifacts" / "e5b" / "e5b_context_crop_002"
    )
    for invalid in ("e5b_context_crop_001", "../e5b_context_crop_002", "e5b_context_crop_latest"):
        with pytest.raises(ValueError, match="实验名称"):
            e5b.resolve_e5b_experiment_root(tmp_path, invalid)


def test_normalized_box_audit_accepts_source_rounding_at_image_boundary() -> None:
    """若冻结源标签的六位小数边界舍入被误报为越界，本测试必须失败。"""

    e5b = _e5b()

    assert e5b.normalized_box_is_in_bounds((0.979688, 0.538889, 0.040625, 0.111111)) is True
    assert e5b.normalized_box_is_in_bounds((0.98, 0.5, 0.04001, 0.1)) is False


def test_training_kwargs_are_an_exact_control_copy_except_output_and_manifest() -> None:
    """若E5-B1训练轮数、增强或优化参数偏离延长训练对照，本测试必须失败。"""

    e5b = _e5b()
    control = {
        "data": "control.yaml",
        "project": "old-project",
        "name": "extended_training_control",
        "device": "0",
        "epochs": 30,
        "resume": False,
        "exist_ok": False,
        "imgsz": 960,
        "mosaic": 1.0,
        "scale": 0.5,
        "seed": 42,
    }

    result = e5b.build_e5b_training_kwargs(
        control,
        data_yaml=Path("e5b.yaml"),
        project_dir=Path("new-project"),
        run_name="e5b_context_crop",
        device="0",
    )

    assert {k: result[k] for k in control if k not in {"data", "project", "name"}} == {
        k: v for k, v in control.items() if k not in {"data", "project", "name"}
    }
    assert result["data"].endswith("e5b.yaml")
    assert result["project"].endswith("new-project")
    assert result["name"] == "e5b_context_crop"


def test_interrupted_training_can_only_continue_from_matching_last_checkpoint(tmp_path: Path) -> None:
    """若续接使用错误权重、缺少优化器状态或轮次不连续，本测试必须失败。"""

    e5b = _e5b()
    run_dir = tmp_path / "e5b_context_crop"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.write_bytes(b"checkpoint")

    request = e5b.validate_e5b_resume_request(
        last_checkpoint=last,
        run_dir=run_dir,
        completed_epochs=5,
        checkpoint_epoch=4,
        optimizer_present=True,
        requested_epochs=30,
    )

    assert request == {
        "last_checkpoint": str(last.resolve()),
        "completed_epochs": 5,
        "resume_from_epoch": 6,
        "requested_epochs": 30,
    }
    with pytest.raises(ValueError, match="最后检查点"):
        e5b.validate_e5b_resume_request(
            last_checkpoint=weights / "best.pt",
            run_dir=run_dir,
            completed_epochs=5,
            checkpoint_epoch=4,
            optimizer_present=True,
            requested_epochs=30,
        )
    with pytest.raises(ValueError, match="优化器"):
        e5b.validate_e5b_resume_request(
            last_checkpoint=last,
            run_dir=run_dir,
            completed_epochs=5,
            checkpoint_epoch=4,
            optimizer_present=False,
            requested_epochs=30,
        )
    with pytest.raises(ValueError, match="轮次"):
        e5b.validate_e5b_resume_request(
            last_checkpoint=last,
            run_dir=run_dir,
            completed_epochs=5,
            checkpoint_epoch=3,
            optimizer_present=True,
            requested_epochs=30,
        )


def test_tiny_metrics_are_reported_for_each_class() -> None:
    """若安全帽与未佩戴安全帽微小目标未分别统计或误检串类，本测试必须失败。"""

    e5b = _e5b()
    records = [
        {
            "ground_truth": [
                {"class_id": 0, "box": [0.0, 0.0, 8.0, 8.0]},
                {"class_id": 1, "box": [20.0, 20.0, 26.0, 26.0]},
            ],
            "predictions": [
                {"class_id": 0, "box": [0.0, 0.0, 8.0, 8.0]},
                {"class_id": 0, "box": [40.0, 40.0, 46.0, 46.0]},
                {"class_id": 1, "box": [60.0, 60.0, 66.0, 66.0]},
            ],
        }
    ]

    metrics = e5b.summarize_e5b_evaluation(records)

    assert metrics["tiny"] == {"tp": 1, "fn": 1, "fp": 2, "precision": 1 / 3, "recall": 0.5, "f1": 0.4}
    assert metrics["tiny_per_class"]["helmet"] == {
        "tp": 1, "fn": 0, "fp": 1, "precision": 0.5, "recall": 1.0, "f1": 2 / 3
    }
    assert metrics["tiny_per_class"]["no_helmet"] == {
        "tp": 0, "fn": 1, "fp": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0
    }


def test_e5b_cli_has_gated_phases_and_no_test_or_resume_switch() -> None:
    """若命令行允许绕过审计、使用test或恢复训练，本测试必须失败。"""

    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "train" / "pipeline_e5b.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    for phase in ("prepare", "audit-augmentations", "train", "continue-training", "evaluate"):
        assert phase in result.stdout
    assert "--experiment-name" in result.stdout
    assert "--primary-target-size" in result.stdout
    assert "--repeated-target-size" in result.stdout
    assert "--split" not in result.stdout
    assert "--resume" not in result.stdout
    assert "test" not in result.stdout.lower()
