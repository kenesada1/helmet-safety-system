#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import shutil
import sys
import time
from typing import TextIO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.e7_lfr import (  # noqa: E402
    E7_LFR_INDEX,
    LiteFeatureRefinement,
    build_resume_plan,
    register_lfr_module,
    restore_early_stopping,
)


OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "e7" / "e7_yolo11s_p2_lfr_001"
LAST_PT = OUTPUT_DIR / "weights" / "last.pt"
BEST_PT = OUTPUT_DIR / "weights" / "best.pt"
RESULTS_CSV = OUTPUT_DIR / "results.csv"
TRAINING_REPORT = OUTPUT_DIR / "e7_training_report.json"
EVALUATION_REPORT = OUTPUT_DIR / "e4_e6_e7_full_val_comparison.json"
PIPELINE_STATUS = OUTPUT_DIR / "e7_pipeline_status.json"
SIDECAR_PREFIX = OUTPUT_DIR.parent / OUTPUT_DIR.name
PREFLIGHT_SIDECAR = Path(f"{SIDECAR_PREFIX}_preflight.json")
TRANSFER_SIDECAR = Path(f"{SIDECAR_PREFIX}_weight_transfer.json")
PARAMETERS_SIDECAR = Path(f"{SIDECAR_PREFIX}_training_parameters.json")
INIT_CHECKPOINT = Path(f"{SIDECAR_PREFIX}_init.pt")
CONSOLE_LOG = Path(f"{SIDECAR_PREFIX}_training.log")
MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11s-p2-lfr.yaml"
MODULE_SOURCE = PROJECT_ROOT / "src" / "helmet_safety" / "training" / "e7_lfr.py"
DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_history() -> tuple[list[dict[str, str]], int]:
    with RESULTS_CSV.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("cannot resume E7 without completed epochs in results.csv")
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return rows, int(best["epoch"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume the interrupted E7 run from its exact last checkpoint.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    import torch
    from ultralytics import YOLO

    args = parse_args()
    for required in (
        OUTPUT_DIR,
        LAST_PT,
        BEST_PT,
        RESULTS_CSV,
        PREFLIGHT_SIDECAR,
        TRANSFER_SIDECAR,
        PARAMETERS_SIDECAR,
        INIT_CHECKPOINT,
        CONSOLE_LOG,
        MODEL_CONFIG,
        MODULE_SOURCE,
        DATASET_YAML,
    ):
        if not required.exists():
            raise FileNotFoundError(required.resolve())
    for forbidden in (TRAINING_REPORT, EVALUATION_REPORT):
        if forbidden.exists():
            raise FileExistsError(f"refusing to resume a completed E7 pipeline: {forbidden.resolve()}")
    if PIPELINE_STATUS.exists():
        status = json.loads(PIPELINE_STATUS.read_text(encoding="utf-8-sig"))
        raise FileExistsError(f"remove or resolve existing pipeline status before resume: {status}")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} is unavailable")

    register_lfr_module()
    rows, best_epoch = read_history()
    checkpoint = torch.load(LAST_PT, map_location="cpu", weights_only=False)
    plan = build_resume_plan(
        checkpoint=checkpoint,
        checkpoint_path=LAST_PT,
        completed_epochs=len(rows),
        best_epoch=best_epoch,
        device=args.device,
        workers=args.workers,
    )
    best_csv_fitness = float(rows[best_epoch - 1]["metrics/mAP50-95(B)"])
    if abs(float(plan["best_fitness"]) - best_csv_fitness) > 1e-6:
        raise RuntimeError(
            f"checkpoint/CSV best fitness mismatch: checkpoint={plan['best_fitness']}, CSV={best_csv_fitness}"
        )
    if args.validate_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return 0

    resumed_at = utc_now()
    resumed_timer = time.perf_counter()
    with CONSOLE_LOG.open("a", encoding="utf-8", buffering=1) as log_handle:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(original_stdout, log_handle)  # type: ignore[assignment]
        sys.stderr = TeeStream(original_stderr, log_handle)  # type: ignore[assignment]
        try:
            print(f"\nE7_RESUMED_AT_UTC={resumed_at}", flush=True)
            print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
            model = YOLO(str(LAST_PT.resolve()), task="detect")
            if [float(value) for value in model.model.stride.detach().cpu().tolist()] != [4.0, 8.0, 16.0, 32.0]:
                raise RuntimeError("resume checkpoint lost E7 detection strides")
            if not isinstance(model.model.model[E7_LFR_INDEX], LiteFeatureRefinement):
                raise RuntimeError("resume checkpoint lost project-local LFR module")
            model.add_callback(
                "on_train_start",
                partial(
                    restore_early_stopping,
                    best_epoch=int(plan["best_epoch"]),
                    best_fitness=float(plan["best_fitness"]),
                ),
            )
            model.train(**plan["train_kwargs"])

            final_rows, _ = read_history()
            if len(final_rows) <= int(plan["completed_epochs"]):
                raise RuntimeError("resume returned without adding a completed epoch")
            for required in (BEST_PT, LAST_PT, RESULTS_CSV):
                if not required.is_file():
                    raise FileNotFoundError(required.resolve())
            for source_path, output_name in (
                (PREFLIGHT_SIDECAR, "e7_preflight_report.json"),
                (TRANSFER_SIDECAR, "e4_to_e7_weight_transfer_report.json"),
                (PARAMETERS_SIDECAR, "e7_training_parameters.json"),
                (MODEL_CONFIG, "yolo11s-p2-lfr.yaml"),
                (MODULE_SOURCE, "e7_lfr_module.py"),
            ):
                shutil.copy2(source_path, OUTPUT_DIR / output_name)
            preflight = json.loads(PREFLIGHT_SIDECAR.read_text(encoding="utf-8"))
            parameters = json.loads(PARAMETERS_SIDECAR.read_text(encoding="utf-8"))
            transfer = json.loads(TRANSFER_SIDECAR.read_text(encoding="utf-8"))
            report = {
                "status": "passed",
                "experiment_id": "E7",
                "resumed": True,
                "resumed_at_utc": resumed_at,
                "ended_at_utc": utc_now(),
                "resume_duration_seconds": time.perf_counter() - resumed_timer,
                "resume_plan": plan,
                "training_parameters": parameters["training_parameters"],
                "requested_initial_batch": 8,
                "actual_batch": 2,
                "nbs": 64,
                "initialization_checkpoint": str(INIT_CHECKPOINT.resolve()),
                "initialization_checkpoint_sha256": sha256(INIT_CHECKPOINT),
                "e4_source_weight": transfer["source_weight"],
                "e4_source_weight_sha256": transfer["source_weight_sha256"],
                "model_parameters": preflight["architecture"]["parameters"],
                "model_gflops_at_960": preflight["architecture"]["gflops_at_960"],
                "completed_epochs": len(final_rows),
                "outputs": {
                    "run_dir": str(OUTPUT_DIR.resolve()),
                    "best_pt": str(BEST_PT.resolve()),
                    "best_pt_sha256": sha256(BEST_PT),
                    "last_pt": str(LAST_PT.resolve()),
                    "last_pt_sha256": sha256(LAST_PT),
                    "results_csv": str(RESULTS_CSV.resolve()),
                    "console_log": str((OUTPUT_DIR / "e7_training.log").resolve()),
                },
                "data_contract": preflight["data_contract"],
                "test_used": False,
            }
            write_json(TRAINING_REPORT, report)
            print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    shutil.copy2(CONSOLE_LOG, OUTPUT_DIR / "e7_training.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
