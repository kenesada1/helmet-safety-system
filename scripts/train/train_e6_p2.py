#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.e6_p2 import (  # noqa: E402
    assert_output_available,
    build_training_kwargs,
    transfer_e4_state,
)


E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11s-p2.yaml"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "e6" / "e6_yolo11s_p2_001"
PREFLIGHT_SIDECAR = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_preflight.json"
INIT_CHECKPOINT = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_init.pt"
EXPECTED_DATASET = Path(r"D:\datasets\SHWD\processed\dataset.yaml")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_leaves(value: object) -> Iterable[object]:
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed E6 YOLO11s-P2 experiment; test is never used.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO

    args = parse_args()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    assert_output_available(OUTPUT_DIR)
    for sidecar in (PREFLIGHT_SIDECAR, INIT_CHECKPOINT):
        if sidecar.exists():
            raise FileExistsError(f"refusing to overwrite existing E6 sidecar: {sidecar.resolve()}")
    for required in (E4_WEIGHT, MODEL_CONFIG, EXPECTED_DATASET):
        if not required.is_file():
            raise FileNotFoundError(required.resolve())
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} is unavailable")

    dataset = yaml.safe_load(EXPECTED_DATASET.read_text(encoding="utf-8"))
    if dataset.get("train") != "images/train" or dataset.get("val") != "images/val":
        raise RuntimeError("dataset train/val contract changed")
    if dataset.get("names") != {0: "helmet", 1: "no_helmet"}:
        raise RuntimeError(f"unexpected class mapping: {dataset.get('names')}")

    source = YOLO(str(E4_WEIGHT.resolve()), task="detect")
    target = YOLO(str(MODEL_CONFIG.resolve()), task="detect")
    source_strides = [float(value) for value in source.model.stride.detach().cpu().tolist()]
    target_strides = [float(value) for value in target.model.stride.detach().cpu().tolist()]
    if source_strides != [8.0, 16.0, 32.0]:
        raise RuntimeError(f"E4 stride contract failed: {source_strides}")
    if target_strides != [4.0, 8.0, 16.0, 32.0]:
        raise RuntimeError(f"E6 stride contract failed: {target_strides}")
    if dict(source.names) != {0: "helmet", 1: "no_helmet"}:
        raise RuntimeError(f"E4 class mapping changed: {source.names}")

    source_state = source.model.state_dict()
    initial_target_state = target.model.state_dict()
    transferred_state, transfer_report = transfer_e4_state(source_state, initial_target_state)
    unexpected_random = [
        name
        for name in transfer_report["random_initialized_tensors"]
        if not name.startswith("model.25.")
        and not name.startswith("model.26.cv2.0.")
        and not name.startswith("model.26.cv3.0.")
    ]
    if unexpected_random:
        raise RuntimeError(f"non-P2 tensors would remain randomly initialized: {unexpected_random}")
    target.model.load_state_dict(transferred_state, strict=True)
    target.model.names = dict(source.names)

    copied_targets = set(transfer_report["exact_tensors"])
    copied_targets.update(item["target"] for item in transfer_report["remapped_tensors"])
    loaded_state = target.model.state_dict()
    if any(not torch.equal(loaded_state[name].cpu(), transferred_state[name].cpu()) for name in copied_targets):
        raise RuntimeError("post-load transfer integrity check failed")
    random_names = transfer_report["random_initialized_tensors"]
    if any(not torch.equal(loaded_state[name].cpu(), initial_target_state[name].cpu()) for name in random_names):
        raise RuntimeError("new P2 random-initialization integrity check failed")

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    target.model.to(device)
    target.model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        free_before, total_memory = torch.cuda.mem_get_info(device)
    else:
        free_before = total_memory = 0
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        forward_output = target.model(torch.zeros((1, 3, 960, 960), device=device))
    leaves = list(tensor_leaves(forward_output))
    if not leaves or not all(torch.isfinite(item).all().item() for item in leaves):
        raise RuntimeError("E6 eval forward returned no tensors or non-finite values")
    eval_shapes = [list(item.shape) for item in leaves]

    # Batch-2 mixed-precision forward/backward is the pre-training VRAM gate.
    target.model.train()
    target.model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        train_output = target.model(torch.zeros((2, 3, 960, 960), device=device))
        train_leaves = list(tensor_leaves(train_output))
        memory_probe_loss = sum(item.float().mean() for item in train_leaves)
    memory_probe_loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    target.model.zero_grad(set_to_none=True)
    target.model.cpu().eval()
    del forward_output, train_output, leaves, train_leaves, memory_probe_loss
    if device.type == "cuda":
        torch.cuda.empty_cache()

    transfer_report.update(
        {
            "source_detect_strides": source_strides,
            "target_detect_strides": target_strides,
            "source_weight": str(E4_WEIGHT.resolve()),
            "source_weight_sha256": sha256(E4_WEIGHT),
            "model_config": str(MODEL_CONFIG.resolve()),
            "new_random_scope": "P2 fusion layers 23-25 plus Detect branch 0; all compatible E4 P3/P4/P5 tensors transferred",
        }
    )
    preflight = {
        "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_total_bytes": int(total_memory),
            "gpu_free_bytes_before_probe": int(free_before),
        },
        "architecture": {
            "e4_detect_strides": source_strides,
            "e6_detect_strides": target_strides,
            "e6_detect_sources": list(target.model.model[-1].f),
            "parameters": sum(parameter.numel() for parameter in target.model.parameters()),
        },
        "weight_transfer": transfer_report,
        "forward_test": {
            "status": "passed",
            "eval_input_shape": [1, 3, 960, 960],
            "eval_output_tensor_shapes": eval_shapes,
            "memory_probe_input_shape": [2, 3, 960, 960],
            "memory_probe_backward": True,
            "peak_allocated_bytes": peak_memory,
            "peak_allocated_gib": peak_memory / 1024**3,
        },
        "data_contract": {
            "dataset_yaml": str(EXPECTED_DATASET.resolve()),
            "train": dataset["train"],
            "val": dataset["val"],
            "test_used": False,
        },
    }
    write_json(PREFLIGHT_SIDECAR, preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)

    target.save(str(INIT_CHECKPOINT.resolve()))
    del source, target
    if device.type == "cuda":
        torch.cuda.empty_cache()
    training_model = YOLO(str(INIT_CHECKPOINT.resolve()), task="detect")
    if [float(value) for value in training_model.model.stride.detach().cpu().tolist()] != target_strides:
        raise RuntimeError("saved E6 initialization checkpoint lost the four detection strides")
    if dict(training_model.names) != {0: "helmet", 1: "no_helmet"}:
        raise RuntimeError("saved E6 initialization checkpoint lost the class mapping")

    training_kwargs = build_training_kwargs(
        data_yaml=EXPECTED_DATASET,
        output_dir=OUTPUT_DIR,
        device=args.device,
        workers=args.workers,
    )
    print("E6_PRECHECK_PASSED_STARTING_TRAINING", flush=True)
    print(json.dumps(training_kwargs, ensure_ascii=False, indent=2), flush=True)
    training_model.train(**training_kwargs)

    best_pt = OUTPUT_DIR / "weights" / "best.pt"
    last_pt = OUTPUT_DIR / "weights" / "last.pt"
    results_csv = OUTPUT_DIR / "results.csv"
    for required in (best_pt, last_pt, results_csv):
        if not required.is_file():
            raise FileNotFoundError(f"training output missing: {required.resolve()}")
    shutil.copy2(PREFLIGHT_SIDECAR, OUTPUT_DIR / "e6_preflight_report.json")
    ended_at = datetime.now(timezone.utc)
    training_report = {
        "status": "passed",
        "experiment_id": "E6",
        "started_at_utc": started_at.isoformat(),
        "ended_at_utc": ended_at.isoformat(),
        "duration_seconds": time.perf_counter() - started,
        "training_parameters": training_kwargs,
        "initialization_checkpoint": str(INIT_CHECKPOINT.resolve()),
        "initialization_checkpoint_sha256": sha256(INIT_CHECKPOINT),
        "e4_source_weight": str(E4_WEIGHT.resolve()),
        "e4_source_weight_sha256": transfer_report["source_weight_sha256"],
        "outputs": {
            "run_dir": str(OUTPUT_DIR.resolve()),
            "best_pt": str(best_pt.resolve()),
            "best_pt_sha256": sha256(best_pt),
            "last_pt": str(last_pt.resolve()),
            "results_csv": str(results_csv.resolve()),
        },
        "test_used": False,
    }
    write_json(OUTPUT_DIR / "e6_training_report.json", training_report)
    print(json.dumps(training_report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
