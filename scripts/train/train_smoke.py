#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import traceback

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.smoke import (  # noqa: E402
    create_smoke_dataset,
    next_run_name,
    validate_training_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a deterministic one-epoch YOLO SHWD smoke test")
    parser.add_argument("--processed", type=Path, default=Path(r"D:\datasets\SHWD\processed"))
    parser.add_argument(
        "--source-data-yaml", type=Path, default=Path(r"D:\datasets\SHWD\processed\dataset.yaml")
    )
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--val-count", type=int, default=24)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as 0")
    parser.add_argument("--run-name", default="smoke_test")
    return parser


def validate_source_config(config_path: Path, processed_root: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = {int(class_id): name for class_id, name in config.get("names", {}).items()}
    if names != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"unexpected dataset classes in {config_path}: {names}")
    configured_root = Path(config["path"]).resolve()
    if configured_root != processed_root.resolve():
        raise ValueError(f"dataset path mismatch: yaml={configured_root}, requested={processed_root.resolve()}")


def environment_report(torch_module: object, ultralytics_module: object) -> dict[str, object]:
    cuda_available = bool(torch_module.cuda.is_available())
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch_module.__version__,
        "ultralytics": ultralytics_module.__version__,
        "cuda_available": cuda_available,
        "torch_cuda": torch_module.version.cuda,
        "gpu_name": torch_module.cuda.get_device_name(0) if cuda_available else None,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    processed_root = args.processed.resolve()
    source_data_yaml = args.source_data_yaml.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    training_dir = artifacts_dir / "training"
    models_dir = artifacts_dir / "models"
    smoke_datasets_dir = artifacts_dir / "smoke_datasets"
    validate_source_config(source_data_yaml, processed_root)

    if args.device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device!r} requested, but torch.cuda.is_available() is False")

    training_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    smoke_datasets_dir.mkdir(parents=True, exist_ok=True)
    run_name = next_run_name(training_dir, args.run_name)
    smoke_dataset_dir = smoke_datasets_dir / run_name
    subset_report = create_smoke_dataset(
        processed_root,
        smoke_dataset_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )
    for split in ("train", "val"):
        split_report = subset_report["splits"][split]
        if sum(split_report["boxes"].values()) <= 0:
            raise ValueError(f"smoke {split} split contains no instances")

    environment = environment_report(torch, ultralytics)
    training_parameters = {
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "cache": False,
        "seed": args.seed,
        "pretrained": True,
        "plots": True,
        "device": device,
        "deterministic": True,
        "project": str(training_dir),
        "name": run_name,
    }
    print(json.dumps({"environment": environment, "training": training_parameters, "subset": subset_report}, indent=2))

    model_path = models_dir / Path(args.model).name
    downloaded_path = Path(attempt_download_asset(str(model_path))).resolve()
    if downloaded_path != model_path.resolve():
        shutil.copy2(downloaded_path, model_path)
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"pretrained model was not downloaded: {model_path}")
    model = YOLO(str(model_path), task="detect")
    previous_cwd = Path.cwd()
    try:
        os.chdir(models_dir)
        model.train(
            data=str(smoke_dataset_dir / "dataset.yaml"),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            cache=False,
            seed=args.seed,
            pretrained=True,
            plots=True,
            device=device,
            deterministic=True,
            project=str(training_dir),
            name=run_name,
            exist_ok=False,
        )
    finally:
        os.chdir(previous_cwd)

    run_dir = Path(model.trainer.save_dir).resolve()
    output_report = validate_training_outputs(run_dir, expected_epochs=args.epochs)
    report = {
        "status": "passed",
        "environment": environment,
        "source_data_yaml": str(source_data_yaml),
        "processed_root": str(processed_root),
        "pretrained_model": str(model_path.resolve()),
        "training_parameters": training_parameters,
        "subset": subset_report,
        "outputs": output_report,
    }
    report_path = run_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "run_dir": str(run_dir), "report": str(report_path)}, indent=2))
    return report


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
