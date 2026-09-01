#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from helmet_safety.training.baseline import (  # noqa: E402
    analyze_training_results,
    dataset_inventory,
    scan_training_log,
    validate_baseline_outputs,
    write_json_report,
)
from helmet_safety.training.resume import (  # noqa: E402
    build_resume_kwargs,
    cumulative_training_seconds,
    validate_checkpoint_metadata,
    validate_resume_run,
    validate_training_completion,
)
from scripts.train.train_baseline import (  # noqa: E402
    TeeStream,
    baseline_reference_snapshot,
    environment_report,
    preserved_run_snapshots,
    pretrained_model_metadata,
    raw_snapshot,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resume the interrupted M4.5 E4 run from its own epoch-43 last.pt; never evaluates test"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001",
    )
    parser.add_argument(
        "--original-log",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "logs" / "m45_yolo11s_e75_960_001.log",
    )
    parser.add_argument(
        "--resume-log",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "logs" / "m45_yolo11s_e75_960_001_resume.log",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started_clock = time.perf_counter()
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO

        if ultralytics.__version__ != "8.4.120":
            raise RuntimeError(f"Ultralytics version drift: {ultralytics.__version__}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 is unavailable")
        run_dir = args.run_dir.resolve()
        original_log = args.original_log.resolve()
        resume_log = args.resume_log.resolve()
        if not original_log.is_file():
            raise FileNotFoundError(f"original E4 log is missing: {original_log}")
        if resume_log.exists():
            raise FileExistsError(f"refusing to overwrite resume log: {resume_log}")
        contract = validate_resume_run(run_dir, expected_completed_epochs=43)
        training_args = contract["args"]
        data_yaml = Path(str(training_args["data"])).resolve()
        processed_root = Path(yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["path"]).resolve()
        inventory = dataset_inventory(processed_root)
        if inventory["train"]["images"] != 5457 or inventory["val"]["images"] != 607:
            raise ValueError("resume requires all 5,457 train and 607 val images")

        artifacts = PROJECT_ROOT / "artifacts"
        training_dir = artifacts / "training"
        raw_root = Path(r"D:\datasets\SHWD\VOC2028")
        preserved_names = [
            "m45_yolo11n_e100_640_001",
            "m45_yolo11n_e50_960_001",
            "m45_yolo11s_e50_640_001",
        ]
        raw_before = raw_snapshot(raw_root)
        baseline_before = baseline_reference_snapshot(training_dir, "baseline_yolo11n_001")
        preserved_before = preserved_run_snapshots(training_dir, preserved_names)
        model = YOLO(str(contract["last_pt"]), task="detect")
        if dict(model.names) != {0: "helmet", 1: "no_helmet"}:
            raise ValueError(f"unexpected checkpoint classes: {model.names}")
        checkpoint_state = validate_checkpoint_metadata(model.ckpt, expected_completed_epochs=43)
        checkpoint_path = Path(str(contract["last_pt"]))
        checkpoint_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        official_model = Path(r"D:\codes\helmet-safety-system\artifacts\models\yolo11s.pt")
        environment = environment_report(torch, ultralytics)
        original_header, _ = json.JSONDecoder().raw_decode(original_log.read_text(encoding="utf-8", errors="replace"))
        resumed_at = datetime.now(timezone.utc).isoformat()

        resume_log.parent.mkdir(parents=True, exist_ok=True)
        with resume_log.open("x", encoding="utf-8", buffering=1) as log_handle:
            with redirect_stdout(TeeStream(sys.stdout, log_handle)), redirect_stderr(TeeStream(sys.stderr, log_handle)), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                print(
                    json.dumps(
                        {
                            "status": "resuming",
                            "milestone": "M4.5",
                            "experiment_id": "E4",
                            "run_dir": str(run_dir),
                            "resume_checkpoint": str(checkpoint_path),
                            "resume_checkpoint_sha256": checkpoint_digest,
                            "completed_epochs_before_resume": 43,
                            "target_epochs": 75,
                            "resume_kwargs": build_resume_kwargs(),
                            "checkpoint_state": checkpoint_state,
                            "test_evaluated": False,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                model.train(**build_resume_kwargs())
                python_warnings = [str(item.message) for item in caught]

        actual_save_dir = Path(model.trainer.save_dir).resolve()
        if actual_save_dir != run_dir:
            raise RuntimeError(f"resume wrote to unexpected directory: {actual_save_dir}")
        outputs = validate_baseline_outputs(run_dir)
        analysis = analyze_training_results(Path(outputs["results_csv"]), requested_epochs=75)
        completion_reason = validate_training_completion(analysis, requested_epochs=75, patience=15)
        with Path(outputs["results_csv"]).open(encoding="utf-8-sig", newline="") as handle:
            result_rows = list(csv.DictReader(handle))
        total_training_seconds = cumulative_training_seconds(result_rows, resume_after_epoch=43)
        original_warning_audit = scan_training_log(original_log)
        resume_warning_audit = scan_training_log(resume_log)
        invalid_warning = any(
            audit[key]
            for audit in (original_warning_audit, resume_warning_audit)
            for key in ("jpeg_auto_repair_warning", "cache_version_warning", "corrupt_data_warning")
        )
        if invalid_warning:
            raise RuntimeError("invalidating JPEG/cache/corrupt-data warning found across E4 logs")
        raw_after = raw_snapshot(raw_root)
        baseline_after = baseline_reference_snapshot(training_dir, "baseline_yolo11n_001")
        preserved_after = preserved_run_snapshots(training_dir, preserved_names)
        if raw_before != raw_after or baseline_before != baseline_after or preserved_before != preserved_after:
            raise RuntimeError("raw data or an E0-E3 training run changed during resume")

        report = {
            "status": "passed",
            "milestone": "M4.5",
            "experiment_id": "E4",
            "baseline_run": "baseline_yolo11n_001",
            "requested_run_name": run_dir.name,
            "actual_run_name": run_dir.name,
            "started_at_utc": original_header["started_at_utc"],
            "resumed_at_utc": resumed_at,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment,
            "data": {
                "dataset_yaml": str(data_yaml),
                "processed_root": str(processed_root),
                "validation_report": str(Path(r"D:\datasets\SHWD\audit\validation_report.json")),
                "validation_status": "passed",
                "inventory": inventory,
            },
            "raw_dataset_snapshot_before": raw_before,
            "raw_dataset_snapshot_after": raw_after,
            "raw_dataset_unchanged": True,
            "baseline_snapshot_before": baseline_before,
            "baseline_snapshot_after": baseline_after,
            "baseline_unchanged": True,
            "preserved_run_snapshots_before": preserved_before,
            "preserved_run_snapshots_after": preserved_after,
            "preserved_runs_unchanged": True,
            "pretrained_model": str(official_model),
            "pretrained_model_metadata": pretrained_model_metadata(official_model),
            "requested_training_protocol": {
                "core_variable": "combined: YOLO11s + imgsz=960 + epochs=75",
                "requested_batch": 2,
                "full_train_participates_each_epoch": True,
                "resume_override_authorized_by_user": True,
            },
            "training_parameters": training_args,
            "actual_batch": 2,
            "parameter_adjustments": [],
            "resume_history": {
                "resume_used": True,
                "resume_scope": "same E4 run after external interruption",
                "resumed_from_other_experiment": False,
                "completed_epochs_before_resume": 43,
                "remaining_epochs_at_resume": 32,
                "checkpoint": str(checkpoint_path),
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": checkpoint_digest,
                "checkpoint_state": checkpoint_state,
            },
            "model_parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            "console_log": str(resume_log),
            "console_logs": [str(original_log), str(resume_log)],
            "duration_seconds": total_training_seconds,
            "resume_wall_seconds": time.perf_counter() - started_clock,
            "training_analysis": analysis,
            "completion_reason": completion_reason,
            "outputs": outputs,
            "python_warnings": python_warnings,
            "warning_audit": {"original": original_warning_audit, "resume": resume_warning_audit},
            "run_dir": str(run_dir),
            "test_used_for_training_or_selection": False,
            "test_evaluated": False,
        }
        report_path = run_dir / "baseline_training_report.json"
        write_json_report(report_path, report)
        print(json.dumps({"status": "passed", "run_dir": str(run_dir), "report": str(report_path)}, indent=2))
        return 0
    except Exception as exc:
        failed_path = PROJECT_ROOT / "artifacts" / "logs" / "m45_yolo11s_e75_960_001_resume_failed_report.json"
        failed = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "test_evaluated": False,
        }
        if not failed_path.exists():
            write_json_report(failed_path, failed)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
