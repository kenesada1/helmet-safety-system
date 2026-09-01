from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_train_script() -> object:
    spec = importlib.util.spec_from_file_location("train_baseline_e4", PROJECT_ROOT / "scripts" / "train" / "train_baseline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e4_training_protocol_records_all_three_combined_variables() -> None:
    """E4 报告若把组合实验误写成单变量实验，本测试必须失败。"""

    train_script = load_train_script()
    args = train_script.build_parser().parse_args(
        [
            "--model",
            str(PROJECT_ROOT / "artifacts" / "models" / "yolo11s.pt"),
            "--milestone",
            "M4.5",
            "--experiment-id",
            "E4",
            "--epochs",
            "75",
            "--imgsz",
            "960",
            "--batch",
            "2",
        ]
    )

    protocol = train_script.requested_training_protocol(args)

    assert protocol["core_variable"] == "combined: YOLO11s + imgsz=960 + epochs=75"
    assert protocol["requested_batch"] == 2


def test_e4_contract_rejects_wrong_epochs_imgsz_or_requested_batch() -> None:
    """E4 评估若接受偏离 75/960/requested batch=2 的训练报告，本测试必须失败。"""

    e4 = importlib.import_module("helmet_safety.training.e4_evaluation")
    valid = {
        "status": "passed",
        "milestone": "M4.5",
        "experiment_id": "E4",
        "baseline_run": "baseline_yolo11n_001",
        "pretrained_model": str(PROJECT_ROOT / "artifacts" / "models" / "yolo11s.pt"),
        "requested_training_protocol": {"requested_batch": 2},
        "training_parameters": {"epochs": 75, "imgsz": 960},
    }

    e4.validate_e4_training_contract(valid)
    for field, value in (("epochs", 74), ("imgsz", 640)):
        invalid = {**valid, "training_parameters": {**valid["training_parameters"], field: value}}
        with pytest.raises(ValueError, match="E4 contract"):
            e4.validate_e4_training_contract(invalid)
    invalid_batch = {**valid, "requested_training_protocol": {"requested_batch": 1}}
    with pytest.raises(ValueError, match="E4 contract"):
        e4.validate_e4_training_contract(invalid_batch)


def test_unified_row_exposes_required_cost_quality_and_difficulty_fields() -> None:
    """统一比较若漏掉成本、困难场景或 FP/FN 字段，本测试必须失败。"""

    e4 = importlib.import_module("helmet_safety.training.e4_evaluation")
    row = e4.build_unified_row(
        experiment="E4",
        model="YOLO11s",
        imgsz=960,
        epochs=75,
        actual_batch=2,
        best_epoch=63,
        early_stopping=False,
        metrics={
            "overall": {"precision": 0.95, "recall": 0.93, "map50": 0.97, "map50_95": 0.66},
            "per_class": {
                "helmet": {"precision": 0.96, "recall": 0.94, "map50": 0.97, "map50_95": 0.78},
                "no_helmet": {"precision": 0.94, "recall": 0.92, "map50": 0.96, "map50_95": 0.54},
            },
        },
        no_helmet_recall_10_30=0.93,
        dense_no_helmet_recall=0.94,
        fp=500,
        fn=600,
        parameters=9_458_752,
        weight_bytes=19_000_000,
        training_seconds=25_000.0,
        inference_ms=12.5,
        throughput=80.0,
        gpu_memory_gib=5.4,
        overfitting=False,
    )

    assert row == {
        "experiment": "E4",
        "model": "YOLO11s",
        "imgsz": 960,
        "epochs": 75,
        "actual_batch": 2,
        "best_epoch": 63,
        "early_stopping": False,
        "overall": {"precision": 0.95, "recall": 0.93, "map50": 0.97, "map50_95": 0.66},
        "helmet": {"precision": 0.96, "recall": 0.94, "map50": 0.97, "map50_95": 0.78},
        "no_helmet": {"precision": 0.94, "recall": 0.92, "map50": 0.96, "map50_95": 0.54},
        "no_helmet_recall_10_30": 0.93,
        "dense_no_helmet_recall": 0.94,
        "fp": 500,
        "fn": 600,
        "parameters": 9_458_752,
        "weight_bytes": 19_000_000,
        "training_seconds": 25_000.0,
        "inference_ms_per_image": 12.5,
        "gpu_throughput_images_per_second": 80.0,
        "gpu_memory_gib": 5.4,
        "overfitting": False,
    }


def test_combined_10_to_30_recall_is_instance_weighted() -> None:
    """10～30 像素 Recall 若错误地取两个分桶的简单平均，本测试必须失败。"""

    e4 = importlib.import_module("helmet_safety.training.e4_evaluation")
    slices = {
        "size_bins": {
            "10_lt_equivalent_size_le_20": {"no_helmet_tp": 90, "no_helmet_fn": 10},
            "20_lt_equivalent_size_le_30": {"no_helmet_tp": 50, "no_helmet_fn": 50},
        }
    }

    assert e4.combined_no_helmet_recall_10_30(slices) == 0.7


def test_e4_metric_deltas_compare_against_every_existing_experiment() -> None:
    """E4 统一结论若只比较 baseline 而漏掉 E1～E3，本测试必须失败。"""

    e4 = importlib.import_module("helmet_safety.training.e4_evaluation")
    rows = [
        {"experiment": "E0", "overall": {"recall": 0.80}, "no_helmet": {"recall": 0.70}},
        {"experiment": "E1", "overall": {"recall": 0.81}, "no_helmet": {"recall": 0.72}},
        {"experiment": "E2", "overall": {"recall": 0.84}, "no_helmet": {"recall": 0.76}},
        {"experiment": "E3", "overall": {"recall": 0.83}, "no_helmet": {"recall": 0.75}},
        {"experiment": "E4", "overall": {"recall": 0.87}, "no_helmet": {"recall": 0.82}},
    ]

    deltas = e4.e4_metric_deltas(rows, scopes=("overall", "no_helmet"), metrics=("recall",))

    assert deltas["vs_E0"]["overall"]["recall"] == pytest.approx(0.07)
    assert deltas["vs_E1"]["no_helmet"]["recall"] == pytest.approx(0.10)
    assert deltas["vs_E2"]["overall"]["recall"] == pytest.approx(0.03)
    assert deltas["vs_E3"]["no_helmet"]["recall"] == pytest.approx(0.07)


def test_e4_evaluator_cli_is_val_only_and_publishes_fixed_rules() -> None:
    """E4 入口若可选择 test 或隐去固定匹配规则，本测试必须失败。"""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate" / "evaluate_e4.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--training-report" in result.stdout
    assert "--split" not in result.stdout
    assert "imgsz=960" in result.stdout
    assert "conf=0.25" in result.stdout
    assert "IoU=0.5" in result.stdout
    assert "E0" in result.stdout and "E4" in result.stdout


def test_resume_contract_accepts_only_the_interrupted_e4_run(tmp_path: Path) -> None:
    """恢复入口若接受错误的 epoch、imgsz、batch 或运行名，本测试必须失败。"""

    resume = importlib.import_module("helmet_safety.training.resume")
    run_dir = tmp_path / "m45_yolo11s_e75_960_001"
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(b"last-checkpoint")
    (weights / "best.pt").write_bytes(b"best-checkpoint")
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump(
            {
                "model": r"D:\codes\helmet-safety-system\artifacts\models\yolo11s.pt",
                "data": r"D:\datasets\SHWD\processed\dataset.yaml",
                "epochs": 75,
                "patience": 15,
                "imgsz": 960,
                "batch": 2,
                "workers": 0,
                "device": "0",
                "seed": 42,
                "deterministic": True,
                "cache": False,
                "amp": True,
                "plots": True,
                "name": "m45_yolo11s_e75_960_001",
                "resume": False,
            }
        ),
        encoding="utf-8",
    )
    rows = ["epoch,time"] + [f"{index},{index * 10}" for index in range(1, 44)]
    (run_dir / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    contract = resume.validate_resume_run(run_dir, expected_completed_epochs=43)

    assert contract["completed_epochs"] == 43
    assert contract["remaining_epochs"] == 32
    assert contract["last_pt"] == str((weights / "last.pt").resolve())
    invalid = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    invalid["batch"] = 1
    (run_dir / "args.yaml").write_text(yaml.safe_dump(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="resume contract"):
        resume.validate_resume_run(run_dir, expected_completed_epochs=43)


def test_resume_checkpoint_requires_optimizer_ema_and_epoch_42() -> None:
    """last.pt 若不含完整优化器/EMA 状态或不是 epoch 43 边界，本测试必须失败。"""

    resume = importlib.import_module("helmet_safety.training.resume")
    checkpoint = {
        "epoch": 42,
        "optimizer": {"state": {}},
        "ema": object(),
        "train_args": {"epochs": 75, "imgsz": 960, "batch": 2},
    }

    state = resume.validate_checkpoint_metadata(checkpoint, expected_completed_epochs=43)

    assert state == {"checkpoint_epoch_zero_based": 42, "target_epochs": 75, "optimizer_present": True, "ema_present": True}
    with pytest.raises(ValueError, match="checkpoint state"):
        resume.validate_checkpoint_metadata({**checkpoint, "optimizer": None}, expected_completed_epochs=43)


def test_resume_call_changes_only_ultralytics_resume_flag() -> None:
    """恢复入口若重新注入学习率、增强或其他超参数，本测试必须失败。"""

    resume = importlib.import_module("helmet_safety.training.resume")

    assert resume.build_resume_kwargs() == {"resume": True}


def test_resume_completion_accepts_target_or_patience_early_stop_only() -> None:
    """恢复训练可正常跑满，也可按 patience 合法早停，但不能接受任意中断。"""

    resume = importlib.import_module("helmet_safety.training.resume")

    assert resume.validate_training_completion(
        {"epochs_completed": 75, "best_epoch": 61, "last_epoch": 75},
        requested_epochs=75,
        patience=15,
    ) == "requested_epochs_completed"
    assert resume.validate_training_completion(
        {"epochs_completed": 58, "best_epoch": 43, "last_epoch": 58},
        requested_epochs=75,
        patience=15,
    ) == "early_stopping_patience_exhausted"
    with pytest.raises(ValueError, match="neither complete nor a valid early stop"):
        resume.validate_training_completion(
            {"epochs_completed": 50, "best_epoch": 43, "last_epoch": 50},
            requested_epochs=75,
            patience=15,
        )


def test_resume_duration_adds_pre_and_post_restart_clocks() -> None:
    """Ultralytics 恢复后 time 会重新计时，报告必须合并两段累计耗时。"""

    resume = importlib.import_module("helmet_safety.training.resume")
    rows = [
        {"epoch": str(epoch), "time": str(epoch * 600.0)} for epoch in range(1, 44)
    ] + [
        {"epoch": "44", "time": "620.0"},
        {"epoch": "45", "time": "1230.0"},
    ]

    assert resume.cumulative_training_seconds(rows, resume_after_epoch=43) == pytest.approx(27030.0)


def test_resume_e4_cli_does_not_offer_test_or_hyperparameter_overrides() -> None:
    """恢复 CLI 若允许 test 或覆盖训练超参数，本测试必须失败。"""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "train" / "resume_e4.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--run-dir" in result.stdout
    for forbidden in ("--split", "--epochs", "--imgsz", "--batch", "--optimizer", "--lr0"):
        assert forbidden not in result.stdout
