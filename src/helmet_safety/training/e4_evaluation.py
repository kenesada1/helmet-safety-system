from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence


def validate_e4_training_contract(training: Mapping[str, object]) -> None:
    expected = {
        "status": "passed",
        "milestone": "M4.5",
        "experiment_id": "E4",
        "baseline_run": "baseline_yolo11n_001",
    }
    parameters = training.get("training_parameters", {})
    protocol = training.get("requested_training_protocol", {})
    model_name = Path(str(training.get("pretrained_model", ""))).name.lower()
    valid = (
        all(training.get(key) == value for key, value in expected.items())
        and isinstance(parameters, Mapping)
        and int(parameters.get("epochs", -1)) == 75
        and int(parameters.get("imgsz", -1)) == 960
        and isinstance(protocol, Mapping)
        and int(protocol.get("requested_batch", -1)) == 2
        and model_name == "yolo11s.pt"
    )
    if not valid:
        raise ValueError("E4 contract requires passed M4.5/E4, YOLO11s, epochs=75, imgsz=960, requested batch=2")


def _quality_metrics(scope: Mapping[str, object]) -> dict[str, float]:
    values = {name: float(scope[name]) for name in ("precision", "recall", "map50", "map50_95")}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("unified comparison metrics must be finite")
    return values


def build_unified_row(
    *,
    experiment: str,
    model: str,
    imgsz: int,
    epochs: int,
    actual_batch: int,
    best_epoch: int,
    early_stopping: bool,
    metrics: Mapping[str, object],
    no_helmet_recall_10_30: float | None,
    dense_no_helmet_recall: float | None,
    fp: int | None,
    fn: int | None,
    parameters: int,
    weight_bytes: int,
    training_seconds: float,
    inference_ms: float,
    throughput: float,
    gpu_memory_gib: float | None,
    overfitting: bool,
) -> dict[str, object]:
    per_class = metrics["per_class"]
    if not isinstance(per_class, Mapping):
        raise ValueError("per_class metrics are required")
    return {
        "experiment": experiment,
        "model": model,
        "imgsz": imgsz,
        "epochs": epochs,
        "actual_batch": actual_batch,
        "best_epoch": best_epoch,
        "early_stopping": early_stopping,
        "overall": _quality_metrics(metrics["overall"]),  # type: ignore[arg-type]
        "helmet": _quality_metrics(per_class["helmet"]),  # type: ignore[arg-type]
        "no_helmet": _quality_metrics(per_class["no_helmet"]),  # type: ignore[arg-type]
        "no_helmet_recall_10_30": no_helmet_recall_10_30,
        "dense_no_helmet_recall": dense_no_helmet_recall,
        "fp": fp,
        "fn": fn,
        "parameters": parameters,
        "weight_bytes": weight_bytes,
        "training_seconds": training_seconds,
        "inference_ms_per_image": inference_ms,
        "gpu_throughput_images_per_second": throughput,
        "gpu_memory_gib": gpu_memory_gib,
        "overfitting": overfitting,
    }


def combined_no_helmet_recall_10_30(slices: Mapping[str, object]) -> float | None:
    size_bins = slices["size_bins"]
    if not isinstance(size_bins, Mapping):
        raise ValueError("size_bins are required")
    tp = fn = 0
    for key in ("10_lt_equivalent_size_le_20", "20_lt_equivalent_size_le_30"):
        row = size_bins[key]
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid size-bin row: {key}")
        tp += int(row["no_helmet_tp"])
        fn += int(row["no_helmet_fn"])
    return tp / (tp + fn) if tp + fn else None


def e4_metric_deltas(
    rows: Sequence[Mapping[str, object]],
    *,
    scopes: Sequence[str] = ("overall", "helmet", "no_helmet"),
    metrics: Sequence[str] = ("precision", "recall", "map50", "map50_95"),
) -> dict[str, object]:
    by_experiment = {str(row["experiment"]): row for row in rows}
    if set(by_experiment) != {"E0", "E1", "E2", "E3", "E4"}:
        raise ValueError("unified comparison requires exactly E0 through E4")
    candidate = by_experiment["E4"]
    result: dict[str, object] = {}
    for experiment in ("E0", "E1", "E2", "E3"):
        reference = by_experiment[experiment]
        result[f"vs_{experiment}"] = {
            scope: {
                metric: float(candidate[scope][metric]) - float(reference[scope][metric])  # type: ignore[index]
                for metric in metrics
            }
            for scope in scopes
        }
    return result
