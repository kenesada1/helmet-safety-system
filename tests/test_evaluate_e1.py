from __future__ import annotations

import importlib
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_val_kwargs_are_val_only_and_preserve_e1_evaluation_contract(tmp_path: Path) -> None:
    """独立 E1 评估必须固定在 val，且不得悄悄改变尺寸、batch 或 worker。"""

    m45 = importlib.import_module("helmet_safety.training.eval_common")

    kwargs = m45.build_val_kwargs(
        data_yaml=tmp_path / "dataset.yaml",
        project_dir=tmp_path / "evaluation",
        run_name="m45_yolo11n_e100_640_val_001",
        batch=8,
        device="0",
    )

    assert kwargs == {
        "data": str(tmp_path / "dataset.yaml"),
        "split": "val",
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "device": "0",
        "plots": True,
        "seed": 42,
        "deterministic": True,
        "project": str(tmp_path / "evaluation"),
        "name": "m45_yolo11n_e100_640_val_001",
        "exist_ok": False,
    }


def test_baseline_comparison_reports_absolute_and_percentage_point_changes() -> None:
    """E1 对比必须逐项使用用户给定的 M4 val 数字，且百分点换算正确。"""

    m45 = importlib.import_module("helmet_safety.training.eval_common")
    e1 = {
        "overall": {"precision": 0.92, "recall": 0.90, "map50": 0.94, "map50_95": 0.61},
        "per_class": {
            "helmet": {"precision": 0.94, "recall": 0.91, "map50": 0.95, "map50_95": 0.74},
            "no_helmet": {"precision": 0.90, "recall": 0.88, "map50": 0.93, "map50_95": 0.48},
        },
    }

    comparison = m45.compare_with_m4_baseline(e1)

    assert comparison["overall"]["recall"] == {
        "m4_baseline": 0.889383,
        "m45_e1": 0.9,
        "absolute_change": 0.010617,
        "percentage_point_change": 1.0617,
    }
    assert comparison["no_helmet"]["map50_95"] == {
        "m4_baseline": 0.471695,
        "m45_e1": 0.48,
        "absolute_change": 0.008305,
        "percentage_point_change": 0.8305,
    }


def test_conclusions_require_safety_gains_without_material_precision_drop() -> None:
    """推荐 100 epochs 必须同时看到安全指标改善，且总体 Precision 不能明显牺牲。"""

    m45 = importlib.import_module("helmet_safety.training.eval_common")
    comparison = {
        "overall": {
            "precision": {"percentage_point_change": -0.4},
            "recall": {"percentage_point_change": 1.1},
        },
        "no_helmet": {
            "recall": {"percentage_point_change": 1.2},
            "map50_95": {"percentage_point_change": 0.8},
        },
    }
    training_analysis = {
        "best_epoch": 67,
        "trailing_window": {"map50_95": {"change": 0.003}},
        "overfitting": {"detected": False},
    }

    conclusions = m45.build_conclusions(comparison, training_analysis)

    assert conclusions == {
        "overall_recall_improved": True,
        "no_helmet_recall_improved": True,
        "no_helmet_map50_95_improved": True,
        "precision_materially_declined": False,
        "best_epoch_beyond_50": True,
        "post_50_state": "plateau",
        "overfitting_detected": False,
        "epochs_100_worthwhile": True,
    }


def test_m45_evaluator_cli_is_val_only() -> None:
    """E1 评估入口不得允许调用者选择 test split。"""

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate" / "evaluate_e1.py"), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--training-report" in result.stdout
    assert "--run-name" in result.stdout
    assert "--split" not in result.stdout
