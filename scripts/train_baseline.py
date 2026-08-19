#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import warnings

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import (  # noqa: E402
    allocate_experiment_name,
    analyze_training_results,
    build_training_kwargs,
    dataset_inventory,
    scan_training_log,
    training_batch_attempts,
    validate_baseline_outputs,
    write_json_report,
)


class TeeStream:
    def __init__(self, terminal: object, log_file: object) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8") or "utf-8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the reproducible full-SHWD YOLO11n M4 baseline")
    parser.add_argument("--data", type=Path, default=Path(r"D:\datasets\SHWD\processed\dataset.yaml"))
    parser.add_argument("--validation-report", type=Path, default=Path(r"D:\datasets\SHWD\audit\validation_report.json"))
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "artifacts" / "models" / "yolo11n.pt")
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--run-name", default="baseline_yolo11n_001")
    parser.add_argument("--milestone", default="M4")
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--baseline-run", default="")
    parser.add_argument(
        "--preserve-run",
        action="append",
        default=[],
        help="Existing training run name to snapshot before and after training; repeatable",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auto-oom-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On CUDA OOM, visibly retry through batch 4→2→1 without changing imgsz",
    )
    return parser


def experiment_metadata(args: argparse.Namespace) -> dict[str, str]:
    return {
        "milestone": args.milestone,
        "experiment_id": args.experiment_id,
        "baseline_run": args.baseline_run,
    }


def requested_training_protocol(args: argparse.Namespace) -> dict[str, object]:
    if args.milestone == "M4.5" and args.experiment_id == "E4":
        core_variable = f"combined: YOLO11s + imgsz={args.imgsz} + epochs={args.epochs}"
    elif args.milestone == "M4.5" and args.experiment_id == "E3":
        core_variable = "model: YOLO11n -> YOLO11s"
    else:
        core_variable = f"imgsz={args.imgsz}"
    return {
        "core_variable": core_variable,
        "requested_batch": args.batch,
        "physical_batch_exception": "physical batch may only decrease after CUDA OOM on the 6GB GPU",
        "full_train_participates_each_epoch": True,
    }


def pretrained_model_metadata(model_path: Path) -> dict[str, object]:
    resolved = model_path.resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": digest.hexdigest()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_report(torch_module: object, ultralytics_module: object) -> dict[str, object]:
    cuda_available = bool(torch_module.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "torch_cuda": torch_module.version.cuda,
        "ultralytics": ultralytics_module.__version__,
        "cuda_available": cuda_available,
        "gpu_name": torch_module.cuda.get_device_name(0) if cuda_available else None,
        "gpu_memory_bytes": torch_module.cuda.get_device_properties(0).total_memory if cuda_available else None,
    }


def raw_snapshot(raw_root: Path) -> dict[str, object]:
    files = [path for path in raw_root.rglob("*") if path.is_file()]
    return {
        "root": str(raw_root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "latest_mtime_ns": max((path.stat().st_mtime_ns for path in files), default=None),
    }


def baseline_reference_snapshot(training_dir: Path, baseline_run: str) -> dict[str, object] | None:
    if not baseline_run:
        return None
    baseline_dir = training_dir / baseline_run
    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"referenced baseline run does not exist: {baseline_dir}")
    return raw_snapshot(baseline_dir)


def preserved_run_snapshots(training_dir: Path, run_names: list[str]) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    for run_name in run_names:
        if run_name in snapshots:
            continue
        run_dir = training_dir / run_name
        if not run_dir.is_dir():
            raise FileNotFoundError(f"preserved training run does not exist: {run_dir}")
        snapshots[run_name] = raw_snapshot(run_dir)
    return snapshots


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"dataset config does not exist: {data_yaml}")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = {int(key): value for key, value in config.get("names", {}).items()}
    if names != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"unexpected class mapping: {names}")
    for split in ("train", "val", "test"):
        if split not in config:
            raise ValueError(f"dataset config is missing {split!r}")
    processed_root = Path(config["path"]).resolve()
    validation = json.loads(args.validation_report.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("issues"):
        raise ValueError(f"dataset validation has not passed: {args.validation_report}")
    inventory = dataset_inventory(processed_root)
    for split in ("train", "val", "test"):
        if inventory[split]["images"] != inventory[split]["labels"] or not inventory[split]["images"]:
            raise ValueError(f"invalid {split} inventory: {inventory[split]}")
    model_path = args.model.resolve()
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"pretrained model does not exist or is empty: {model_path}")
    return data_yaml, processed_root, validation, inventory


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "cuda" in text and ("out of memory" in text or "cuda oom" in text)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    data_yaml, processed_root, validation, inventory = validate_inputs(args)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} was requested but CUDA is unavailable")
    if args.imgsz != 640:
        print(f"WARNING: non-baseline imgsz explicitly requested: {args.imgsz}")
    artifacts_dir = args.artifacts_dir.resolve()
    training_dir = artifacts_dir / "training"
    logs_dir = artifacts_dir / "logs"
    training_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_report(torch, ultralytics)
    pretrained_metadata = pretrained_model_metadata(args.model)
    raw_before = raw_snapshot(Path(validation["raw_root"]))
    baseline_before = baseline_reference_snapshot(training_dir, args.baseline_run)
    preserved_before = preserved_run_snapshots(training_dir, args.preserve_run)
    batches = training_batch_attempts(args.batch, auto_oom_retry=args.auto_oom_retry)
    adjustments: list[dict[str, object]] = []
    attempt_reports: list[str] = []

    for batch in batches:
        run_name = allocate_experiment_name(training_dir, logs_dir, args.run_name)
        log_path = logs_dir / f"{run_name}.log"
        started_at = utc_now()
        started_clock = time.perf_counter()
        run_dir = training_dir / run_name
        kwargs = build_training_kwargs(
            data_yaml=data_yaml,
            project_dir=training_dir,
            run_name=run_name,
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=batch,
            workers=args.workers,
            device=args.device,
            seed=args.seed,
        )
        report: dict[str, object] = {
            "status": "running",
            **experiment_metadata(args),
            "requested_run_name": args.run_name,
            "actual_run_name": run_name,
            "started_at_utc": started_at,
            "environment": environment,
            "data": {
                "dataset_yaml": str(data_yaml),
                "processed_root": str(processed_root),
                "validation_report": str(args.validation_report.resolve()),
                "validation_status": validation["status"],
                "inventory": inventory,
            },
            "raw_dataset_snapshot_before": raw_before,
            "baseline_snapshot_before": baseline_before,
            "preserved_run_snapshots_before": preserved_before,
            "pretrained_model": str(args.model.resolve()),
            "pretrained_model_metadata": pretrained_metadata,
            "requested_training_protocol": requested_training_protocol(args),
            "training_parameters": {"model": str(args.model.resolve()), **kwargs},
            "actual_batch": batch,
            "parameter_adjustments": list(adjustments),
            "console_log": str(log_path.resolve()),
            "test_used_for_training_or_selection": False,
            "test_evaluated": False,
        }
        try:
            with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
                stdout_tee = TeeStream(sys.stdout, log_handle)
                stderr_tee = TeeStream(sys.stderr, log_handle)
                with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee), warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    model = YOLO(str(args.model.resolve()), task="detect")
                    report["model_parameters"] = sum(parameter.numel() for parameter in model.model.parameters())
                    model.train(**kwargs)
                    run_dir = Path(model.trainer.save_dir).resolve()
                    python_warnings = [str(item.message) for item in caught]
            outputs = validate_baseline_outputs(run_dir)
            analysis = analyze_training_results(Path(outputs["results_csv"]), requested_epochs=args.epochs)
            warning_audit = scan_training_log(log_path)
            raw_after = raw_snapshot(Path(validation["raw_root"]))
            baseline_after = baseline_reference_snapshot(training_dir, args.baseline_run)
            preserved_after = preserved_run_snapshots(training_dir, args.preserve_run)
            report.update(
                {
                    "status": "passed",
                    "ended_at_utc": utc_now(),
                    "duration_seconds": time.perf_counter() - started_clock,
                    "training_analysis": analysis,
                    "outputs": outputs,
                    "python_warnings": python_warnings,
                    "warning_audit": warning_audit,
                    "raw_dataset_snapshot_after": raw_after,
                    "raw_dataset_unchanged": raw_before == raw_after,
                    "baseline_snapshot_after": baseline_after,
                    "baseline_unchanged": baseline_before == baseline_after,
                    "preserved_run_snapshots_after": preserved_after,
                    "preserved_runs_unchanged": preserved_before == preserved_after,
                    "run_dir": str(run_dir),
                }
            )
            if warning_audit["jpeg_auto_repair_warning"] or warning_audit["cache_version_warning"]:
                raise RuntimeError(f"invalidating data/cache warning found in {log_path}")
            if raw_before != raw_after:
                raise RuntimeError("raw SHWD metadata changed during training")
            if baseline_before != baseline_after:
                raise RuntimeError("referenced baseline run changed during training")
            if preserved_before != preserved_after:
                raise RuntimeError("a preserved training run changed during training")
            report_path = run_dir / "baseline_training_report.json"
            write_json_report(report_path, report)
            print(json.dumps({"status": "passed", "run_dir": str(run_dir), "report": str(report_path)}, indent=2))
            return report
        except Exception as exc:
            duration = time.perf_counter() - started_clock
            oom = is_cuda_oom(exc)
            report.update(
                {
                    "status": "failed",
                    "ended_at_utc": utc_now(),
                    "duration_seconds": duration,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "cuda_oom": oom,
                    "traceback": traceback.format_exc(),
                }
            )
            failed_report = logs_dir / f"{run_name}_failed_report.json"
            write_json_report(failed_report, report)
            attempt_reports.append(str(failed_report.resolve()))
            if oom and batch != batches[-1]:
                next_batch = batches[batches.index(batch) + 1]
                adjustment = {
                    "reason": "CUDA out of memory",
                    "from_batch": batch,
                    "to_batch": next_batch,
                    "imgsz_unchanged": args.imgsz,
                    "failed_attempt_report": str(failed_report.resolve()),
                }
                adjustments.append(adjustment)
                print(f"CUDA OOM at batch={batch}; explicitly retrying batch={next_batch}, imgsz={args.imgsz}")
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError(f"all training attempts failed: {attempt_reports}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
