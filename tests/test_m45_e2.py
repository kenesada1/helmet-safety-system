from __future__ import annotations

import importlib
import importlib.util
import gc
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_train_script() -> object:
    spec = importlib.util.spec_from_file_location("train_baseline_e2", PROJECT_ROOT / "scripts" / "train_baseline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2_oom_attempts_fall_back_from_batch_four_through_one() -> None:
    """E2 在 batch=2 仍 OOM 时必须继续尝试 batch=1，且禁用重试时只尝试请求值。"""

    baseline = importlib.import_module("helmet_safety.training.baseline")

    assert baseline.training_batch_attempts(4, auto_oom_retry=True) == [4, 2, 1]
    assert baseline.training_batch_attempts(4, auto_oom_retry=False) == [4]


def test_e2_training_protocol_records_requested_batch_and_preserved_e1_run(tmp_path: Path) -> None:
    """训练报告必须区分请求/实际 batch，并能审计 E1 在训练前后未变化。"""

    train_script = load_train_script()
    args = train_script.build_parser().parse_args(
        [
            "--imgsz",
            "960",
            "--batch",
            "4",
            "--preserve-run",
            "m45_yolo11n_e100_640_001",
        ]
    )
    training_dir = tmp_path / "training"
    e1_dir = training_dir / "m45_yolo11n_e100_640_001"
    e1_dir.mkdir(parents=True)
    (e1_dir / "weights.pt").write_bytes(b"unchanged")

    protocol = train_script.requested_training_protocol(args)
    snapshots = train_script.preserved_run_snapshots(training_dir, args.preserve_run)

    assert protocol == {
        "core_variable": "imgsz=960",
        "requested_batch": 4,
        "physical_batch_exception": "physical batch may only decrease after CUDA OOM on the 6GB GPU",
        "full_train_participates_each_epoch": True,
    }
    assert set(snapshots) == {"m45_yolo11n_e100_640_001"}
    assert snapshots["m45_yolo11n_e100_640_001"]["file_count"] == 1
    assert snapshots["m45_yolo11n_e100_640_001"]["total_bytes"] == 9


def test_equivalent_size_buckets_use_original_pixel_box_dimensions() -> None:
    """尺寸层级必须按原图像素框的 sqrt(width*height) 和闭区间边界划分。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")

    cases = [
        ([0.0, 0.0, 10.0, 10.0], "equivalent_size_le_10"),
        ([0.0, 0.0, 20.0, 20.0], "10_lt_equivalent_size_le_20"),
        ([0.0, 0.0, 30.0, 30.0], "20_lt_equivalent_size_le_30"),
        ([0.0, 0.0, 50.0, 50.0], "30_lt_equivalent_size_le_50"),
        ([0.0, 0.0, 51.0, 51.0], "equivalent_size_gt_50"),
        ([5.0, 8.0, 25.0, 13.0], "equivalent_size_le_10"),
    ]

    for box, expected in cases:
        assert e2.size_bucket_for_box(box) == expected


def test_slice_summary_uses_class_aware_iou_matching_and_counts_images() -> None:
    """同位置错类别预测不得成为 TP，且一个预测不能重复匹配多个真实框。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    records = [
        {
            "image_id": "first.jpg",
            "ground_truth": [
                {"class_id": 0, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [20.0, 0.0, 35.0, 15.0]},
                {"class_id": 1, "box": [40.0, 0.0, 55.0, 15.0]},
            ],
            "predictions": [
                {"class_id": 1, "box": [0.0, 0.0, 10.0, 10.0]},
                {"class_id": 1, "box": [20.0, 0.0, 35.0, 15.0]},
            ],
        },
        {
            "image_id": "second.jpg",
            "ground_truth": [
                {"class_id": 0, "box": [0.0, 0.0, 8.0, 8.0]},
            ],
            "predictions": [
                {"class_id": 0, "box": [0.0, 0.0, 8.0, 8.0]},
            ],
        },
    ]

    summary = e2.summarize_detection_slices(records, iou_threshold=0.5)

    tiny = summary["size_bins"]["equivalent_size_le_10"]
    assert tiny == {
        "images": 2,
        "ground_truth_instances": 2,
        "helmet_instances": 2,
        "no_helmet_instances": 0,
        "tp": 1,
        "fn": 1,
        "recall": 0.5,
        "helmet_tp": 1,
        "helmet_fn": 1,
        "helmet_recall": 0.5,
        "no_helmet_tp": 0,
        "no_helmet_fn": 0,
        "no_helmet_recall": None,
    }
    small = summary["size_bins"]["10_lt_equivalent_size_le_20"]
    assert small["images"] == 1
    assert small["ground_truth_instances"] == 2
    assert small["no_helmet_instances"] == 2
    assert small["no_helmet_tp"] == 1
    assert small["no_helmet_fn"] == 1
    assert small["no_helmet_recall"] == 0.5
    assert summary["matching"] == {"class_aware": True, "iou_threshold": 0.5}


def test_dense_scene_summary_reports_threshold_ten_and_twenty_by_class() -> None:
    """密集切片必须分别统计每图真实框 >=10 与 >=20 的 overall/分类 Recall。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    ground_truth = [
        {"class_id": 0 if index < 4 else 1, "box": [index * 20.0, 0.0, index * 20.0 + 12.0, 12.0]}
        for index in range(20)
    ]
    predictions = [dict(item) for item in ground_truth[:3]] + [dict(item) for item in ground_truth[4:14]]

    summary = e2.summarize_detection_slices(
        [{"image_id": "dense.jpg", "ground_truth": ground_truth, "predictions": predictions}],
        iou_threshold=0.5,
    )

    for key in ("ground_truth_gte_10", "ground_truth_gte_20"):
        dense = summary["dense_scenes"][key]
        assert dense["images"] == 1
        assert dense["ground_truth_instances"] == 20
        assert dense["tp"] == 13
        assert dense["fn"] == 7
        assert dense["recall"] == 0.65
        assert dense["helmet_recall"] == 0.75
        assert dense["no_helmet_recall"] == 0.625


def test_e2_comparison_uses_e2_key_and_percentage_points() -> None:
    """E2 主指标报告不得沿用 E1 键名，且变化值必须同时保留绝对值和百分点。"""

    m45 = importlib.import_module("helmet_safety.training.m45")
    metrics = {
        "overall": {"precision": 0.90, "recall": 0.91, "map50": 0.94, "map50_95": 0.62},
        "per_class": {
            "helmet": {"precision": 0.95, "recall": 0.92, "map50": 0.96, "map50_95": 0.75},
            "no_helmet": {"precision": 0.85, "recall": 0.90, "map50": 0.92, "map50_95": 0.49},
        },
    }

    comparison = m45.compare_candidate_with_m4_baseline(metrics, candidate_key="m45_e2")

    assert comparison["overall"]["recall"] == {
        "m4_baseline": 0.889383,
        "m45_e2": 0.91,
        "absolute_change": 0.020617,
        "percentage_point_change": 2.0617,
    }
    assert "m45_e1" not in comparison["overall"]["recall"]


def test_slice_comparison_reports_fixed_baseline_to_e2_recall_changes() -> None:
    """尺寸与密集切片必须逐项使用同口径 baseline/E2 Recall 计算变化。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    baseline = {
        "size_bins": {"10_lt_equivalent_size_le_20": {"recall": 0.4, "helmet_recall": 0.5, "no_helmet_recall": 0.3}},
        "dense_scenes": {"ground_truth_gte_10": {"recall": 0.7, "helmet_recall": 0.8, "no_helmet_recall": 0.6}},
    }
    candidate = {
        "size_bins": {"10_lt_equivalent_size_le_20": {"recall": 0.5, "helmet_recall": 0.55, "no_helmet_recall": 0.45}},
        "dense_scenes": {"ground_truth_gte_10": {"recall": 0.75, "helmet_recall": 0.81, "no_helmet_recall": 0.65}},
    }

    comparison = e2.compare_slice_recalls(baseline, candidate)

    assert comparison["size_bins"]["10_lt_equivalent_size_le_20"]["no_helmet_recall"] == {
        "m4_baseline_640": 0.3,
        "m45_e2_960": 0.45,
        "absolute_change": 0.15,
        "percentage_point_change": 15.0,
    }
    assert comparison["dense_scenes"]["ground_truth_gte_10"]["recall"]["percentage_point_change"] == 5.0


def test_reported_gpu_memory_uses_largest_ultralytics_gib_sample(tmp_path: Path) -> None:
    """显存字段应取训练日志中的最大 GiB 样本，并明确它是日志报告值。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    log_path = tmp_path / "train.log"
    log_path.write_text("GPU_mem 1.27G\nepoch 2.31G\nepoch 2.08G\n", encoding="utf-8")

    assert e2.reported_gpu_memory(log_path) == {
        "source": "maximum GiB value reported in the Ultralytics training log",
        "max_reported_gib": 2.31,
        "max_reported_bytes": 2480343613,
    }


def test_model_stages_release_each_model_before_loading_the_next() -> None:
    """6GB GPU 的 E2/last/baseline 评估必须顺序加载，不能同时保留多个模型。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    state = {"live": 0, "maximum": 0, "cleanups": 0}

    class TrackedModel:
        def __init__(self, name: str) -> None:
            self.name = name
            state["live"] += 1
            state["maximum"] = max(state["maximum"], state["live"])

        def __del__(self) -> None:
            state["live"] -= 1

    def loader(path: Path) -> TrackedModel:
        return TrackedModel(path.stem)

    def runner(name: str, model: TrackedModel) -> str:
        return f"{name}:{model.name}"

    def cleanup() -> None:
        gc.collect()
        state["cleanups"] += 1

    results = e2.run_sequential_model_stages(
        [("e2", Path("e2.pt")), ("last", Path("last.pt")), ("baseline", Path("baseline.pt"))],
        loader=loader,
        runner=runner,
        cleanup=cleanup,
    )

    assert results == {"e2": "e2:e2", "last": "last:last", "baseline": "baseline:baseline"}
    assert state == {"live": 0, "maximum": 1, "cleanups": 3}


def test_val_prediction_source_streams_the_directory_instead_of_materializing_all_paths(tmp_path: Path) -> None:
    """607 张 val 必须通过目录数据源按 batch 流式读取，不能作为整张图片列表一次性物化。"""

    e2 = importlib.import_module("helmet_safety.training.m45_e2")
    images_dir = tmp_path / "images" / "val"
    images_dir.mkdir(parents=True)

    source = e2.streaming_image_source(images_dir)

    assert source == str(images_dir.resolve())
    assert not isinstance(source, list)


def test_e2_evaluator_cli_is_val_only_and_fixes_matching_thresholds() -> None:
    """E2 入口不得提供 test split，且固定展示 conf=0.25 与 matching IoU=0.5。"""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_m45_e2.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--training-report" in result.stdout
    assert "--split" not in result.stdout
    assert "0.25" in result.stdout
    assert "0.5" in result.stdout
