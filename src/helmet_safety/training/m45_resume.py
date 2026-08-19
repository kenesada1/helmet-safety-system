from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import yaml


def validate_resume_run(run_dir: Path, *, expected_completed_epochs: int) -> dict[str, object]:
    resolved = run_dir.resolve()
    args_path = resolved / "args.yaml"
    results_path = resolved / "results.csv"
    last_pt = resolved / "weights" / "last.pt"
    best_pt = resolved / "weights" / "best.pt"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (args_path, results_path, last_pt, best_pt)):
        raise ValueError("resume contract requires args.yaml, results.csv, best.pt, and last.pt")
    arguments = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    required = {
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
    model_name = Path(str(arguments.get("model", ""))).name.lower()
    with results_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    valid = (
        resolved.name == "m45_yolo11s_e75_960_001"
        and model_name == "yolo11s.pt"
        and all(arguments.get(key) == value for key, value in required.items())
        and len(rows) == expected_completed_epochs
        and rows
        and int(float(rows[-1]["epoch"])) == expected_completed_epochs
        and expected_completed_epochs < 75
    )
    if not valid:
        raise ValueError("resume contract does not match interrupted M4.5 E4 at the expected epoch boundary")
    return {
        "run_dir": str(resolved),
        "args": arguments,
        "completed_epochs": expected_completed_epochs,
        "remaining_epochs": 75 - expected_completed_epochs,
        "last_pt": str(last_pt.resolve()),
        "best_pt": str(best_pt.resolve()),
        "results_csv": str(results_path.resolve()),
    }


def validate_checkpoint_metadata(
    checkpoint: Mapping[str, object], *, expected_completed_epochs: int
) -> dict[str, object]:
    train_args = checkpoint.get("train_args")
    valid = (
        isinstance(train_args, Mapping)
        and int(checkpoint.get("epoch", -1)) == expected_completed_epochs - 1
        and checkpoint.get("optimizer") is not None
        and checkpoint.get("ema") is not None
        and int(train_args.get("epochs", -1)) == 75
        and int(train_args.get("imgsz", -1)) == 960
        and int(train_args.get("batch", -1)) == 2
    )
    if not valid:
        raise ValueError("checkpoint state is incomplete or does not match E4 epoch 43")
    return {
        "checkpoint_epoch_zero_based": expected_completed_epochs - 1,
        "target_epochs": 75,
        "optimizer_present": True,
        "ema_present": True,
    }


def build_resume_kwargs() -> dict[str, bool]:
    return {"resume": True}


def validate_training_completion(
    analysis: Mapping[str, object], *, requested_epochs: int, patience: int
) -> str:
    """Accept a full run or an Ultralytics patience-bound early stop."""

    completed = int(analysis.get("epochs_completed", -1))
    best_epoch = int(analysis.get("best_epoch", -1))
    last_epoch = int(analysis.get("last_epoch", -1))
    if completed == requested_epochs and last_epoch == requested_epochs:
        return "requested_epochs_completed"
    if (
        0 < completed < requested_epochs
        and last_epoch == completed
        and best_epoch > 0
        and last_epoch - best_epoch >= patience
    ):
        return "early_stopping_patience_exhausted"
    raise ValueError(
        "resumed training is neither complete nor a valid early stop: "
        f"completed={completed}, best_epoch={best_epoch}, last_epoch={last_epoch}, "
        f"requested_epochs={requested_epochs}, patience={patience}"
    )


def cumulative_training_seconds(
    rows: Sequence[Mapping[str, object]], *, resume_after_epoch: int
) -> float:
    """Merge Ultralytics' pre-resume and reset post-resume cumulative clocks."""

    if not rows:
        raise ValueError("training duration requires at least one result row")
    parsed = [(int(float(row["epoch"])), float(row["time"])) for row in rows]
    if not all(math.isfinite(seconds) and seconds >= 0 for _, seconds in parsed):
        raise ValueError("training duration requires finite non-negative times")
    final_epoch, final_seconds = parsed[-1]
    if final_epoch <= resume_after_epoch:
        return final_seconds
    boundary = next((seconds for epoch, seconds in parsed if epoch == resume_after_epoch), None)
    if boundary is None:
        raise ValueError(f"resume boundary epoch {resume_after_epoch} is missing")
    first_post = next(seconds for epoch, seconds in parsed if epoch > resume_after_epoch)
    return boundary + final_seconds if first_post < boundary else final_seconds
