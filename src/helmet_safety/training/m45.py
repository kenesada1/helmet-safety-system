from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping


M4_BASELINE_VAL = {
    "overall": {"precision": 0.928870, "recall": 0.889383, "map50": 0.938299, "map50_95": 0.603832},
    "helmet": {"precision": 0.943917, "recall": 0.908969, "map50": 0.949804, "map50_95": 0.735969},
    "no_helmet": {"precision": 0.913823, "recall": 0.869797, "map50": 0.926793, "map50_95": 0.471695},
}


def build_val_kwargs(
    *,
    data_yaml: Path,
    project_dir: Path,
    run_name: str,
    imgsz: int = 640,
    batch: int = 8,
    workers: int = 0,
    device: str = "0",
    seed: int = 42,
) -> dict[str, object]:
    """Build the val-only contract for the M4.5 E1 best-weight evaluation."""

    return {
        "data": str(data_yaml),
        "split": "val",
        "imgsz": imgsz,
        "batch": batch,
        "workers": workers,
        "device": device,
        "plots": True,
        "seed": seed,
        "deterministic": True,
        "project": str(project_dir),
        "name": run_name,
        "exist_ok": False,
    }


def compare_candidate_with_m4_baseline(
    candidate_metrics: Mapping[str, object], *, candidate_key: str
) -> dict[str, object]:
    """Compare one M4.5 val result with the fixed, user-provided M4 reference."""

    if not candidate_key:
        raise ValueError("candidate key must not be empty")
    candidate_scopes = {
        "overall": candidate_metrics["overall"],
        "helmet": candidate_metrics["per_class"]["helmet"],  # type: ignore[index]
        "no_helmet": candidate_metrics["per_class"]["no_helmet"],  # type: ignore[index]
    }
    comparison: dict[str, object] = {}
    for scope, baseline_values in M4_BASELINE_VAL.items():
        candidate_values = candidate_scopes[scope]
        scope_result: dict[str, object] = {}
        for metric, baseline_value in baseline_values.items():
            candidate_value = float(candidate_values[metric])  # type: ignore[index]
            if not math.isfinite(candidate_value):
                raise ValueError(f"non-finite candidate metric: {scope}.{metric}={candidate_value}")
            change = round(candidate_value - baseline_value, 6)
            scope_result[metric] = {
                "m4_baseline": baseline_value,
                candidate_key: candidate_value,
                "absolute_change": change,
                "percentage_point_change": round(change * 100, 4),
            }
        comparison[scope] = scope_result
    return comparison


def compare_with_m4_baseline(e1_metrics: Mapping[str, object]) -> dict[str, object]:
    """Backward-compatible E1 comparison wrapper."""

    return compare_candidate_with_m4_baseline(e1_metrics, candidate_key="m45_e1")


def build_conclusions(
    comparison: Mapping[str, object], training_analysis: Mapping[str, object]
) -> dict[str, object]:
    overall = comparison["overall"]  # type: ignore[assignment]
    no_helmet = comparison["no_helmet"]  # type: ignore[assignment]
    overall_recall_improved = overall["recall"]["percentage_point_change"] > 0  # type: ignore[index]
    no_helmet_recall_improved = no_helmet["recall"]["percentage_point_change"] > 0  # type: ignore[index]
    no_helmet_map_improved = no_helmet["map50_95"]["percentage_point_change"] > 0  # type: ignore[index]
    precision_materially_declined = overall["precision"]["percentage_point_change"] < -1.0  # type: ignore[index]
    best_beyond_50 = int(training_analysis["best_epoch"]) > 50
    trailing_map_change = float(training_analysis["trailing_window"]["map50_95"]["change"])  # type: ignore[index]
    post_50_state = "continuing_improvement" if best_beyond_50 and trailing_map_change > 0.005 else "plateau"
    overfitting = bool(training_analysis["overfitting"]["detected"])  # type: ignore[index]
    return {
        "overall_recall_improved": overall_recall_improved,
        "no_helmet_recall_improved": no_helmet_recall_improved,
        "no_helmet_map50_95_improved": no_helmet_map_improved,
        "precision_materially_declined": precision_materially_declined,
        "best_epoch_beyond_50": best_beyond_50,
        "post_50_state": post_50_state,
        "overfitting_detected": overfitting,
        "epochs_100_worthwhile": all(
            (overall_recall_improved, no_helmet_recall_improved, no_helmet_map_improved, best_beyond_50)
        )
        and not precision_materially_declined
        and not overfitting,
    }
