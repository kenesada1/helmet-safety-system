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
from typing import Iterable, TextIO

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.e7_lfr import (  # noqa: E402
    E7_DETECT_INDEX,
    E7_LFR_INDEX,
    LiteFeatureRefinement,
    assert_output_available,
    build_training_kwargs,
    configure_probe_hyperparameters,
    fits_physical_vram,
    register_lfr_module,
    transfer_e4_state,
)


E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E6_CONFIG = PROJECT_ROOT / "configs" / "yolo11s-p2.yaml"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11s-p2-lfr.yaml"
MODULE_SOURCE = PROJECT_ROOT / "src" / "helmet_safety" / "training" / "e7_lfr.py"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "e7" / "e7_yolo11s_p2_lfr_001"
SIDECAR_PREFIX = OUTPUT_DIR.parent / OUTPUT_DIR.name
PREFLIGHT_SIDECAR = Path(f"{SIDECAR_PREFIX}_preflight.json")
TRANSFER_SIDECAR = Path(f"{SIDECAR_PREFIX}_weight_transfer.json")
PARAMETERS_SIDECAR = Path(f"{SIDECAR_PREFIX}_training_parameters.json")
INIT_CHECKPOINT = Path(f"{SIDECAR_PREFIX}_init.pt")
CONSOLE_LOG = Path(f"{SIDECAR_PREFIX}_training.log")
DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
EXPECTED_CLASSES = {0: "helmet", 1: "no_helmet"}
EXPECTED_SPLIT_COUNTS = {"train": 5457, "val": 607}
PROBE_BATCHES = (8, 4, 2)


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def image_names(directory: Path) -> set[str]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {path.name for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions}


def validate_dataset_contract() -> dict[str, object]:
    payload = yaml.safe_load(DATASET_YAML.read_text(encoding="utf-8"))
    if payload.get("train") != "images/train" or payload.get("val") != "images/val":
        raise RuntimeError("dataset train/val contract changed")
    if payload.get("names") != EXPECTED_CLASSES:
        raise RuntimeError(f"unexpected class mapping: {payload.get('names')}")
    root = Path(str(payload["path"])).resolve()
    split_names = {split: image_names(root / "images" / split) for split in ("train", "val", "test")}
    split_counts = {split: len(names) for split, names in split_names.items()}
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        if split_counts[split] != expected:
            raise RuntimeError(f"{split} image count changed: {split_counts[split]} != {expected}")
    overlap = sorted(split_names["train"] & split_names["val"])
    if overlap:
        raise RuntimeError(f"train/val image leakage detected: {overlap[:20]}")
    return {
        "dataset_yaml": str(DATASET_YAML.resolve()),
        "dataset_root": str(root),
        "train": payload["train"],
        "val": payload["val"],
        "test_declared_but_unused": payload.get("test"),
        "split_image_counts": split_counts,
        "train_val_filename_overlap": 0,
        "external_data_used": False,
        "e5_data_used": False,
        "resampling_used": False,
        "context_crop_used": False,
        "sliding_window_used": False,
        "test_used": False,
    }


def synthetic_detection_batch(batch_size: int, device: object) -> dict[str, object]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(42 + batch_size)
    images = torch.rand((batch_size, 3, 960, 960), generator=generator).to(device, non_blocking=False)
    return {
        "img": images,
        "batch_idx": torch.arange(batch_size, device=device, dtype=torch.float32),
        "cls": torch.tensor([index % 2 for index in range(batch_size)], device=device, dtype=torch.float32).view(-1, 1),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.08, 0.08] for _ in range(batch_size)], device=device, dtype=torch.float32
        ),
    }


def is_cuda_oom(error: BaseException) -> bool:
    import torch

    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def run_batch_probe(
    *, batch_size: int, initial_state: dict[str, object], device: object
) -> dict[str, object]:
    import torch
    from ultralytics import YOLO

    register_lfr_module()
    probe = YOLO(str(MODEL_CONFIG.resolve()), task="detect")
    probe.model.load_state_dict(initial_state, strict=True)
    probe.model.names = dict(EXPECTED_CLASSES)
    network = probe.model.to(device).train()
    configure_probe_hyperparameters(network)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        free_before, total_memory = torch.cuda.mem_get_info(device)
    else:
        free_before = total_memory = 0
    started = time.perf_counter()
    last_loss_items: dict[str, float] = {}
    try:
        # Two full optimizer iterations exercise forward, real YOLO detection loss,
        # backward, gradients and AdamW state allocation rather than a proxy mean.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            batch = synthetic_detection_batch(batch_size, device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, loss_items = network.loss(batch)
                scalar_loss = loss.sum()
            if not torch.isfinite(scalar_loss).item():
                raise RuntimeError(f"non-finite probe loss for batch {batch_size}")
            scalar_loss.backward()
            if not any(parameter.grad is not None for parameter in network.parameters()):
                raise RuntimeError(f"no gradients produced for batch {batch_size}")
            optimizer.step()
            last_loss_items = {key: float(value) for key, value in loss_items.items()}
            del batch, loss, loss_items, scalar_loss
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak = int(torch.cuda.max_memory_allocated(device))
        else:
            peak = 0
        physical_capacity_passed = device.type != "cuda" or fits_physical_vram(
            peak_allocated_bytes=peak, total_bytes=int(total_memory)
        )
        return {
            "batch": batch_size,
            "status": "passed" if physical_capacity_passed else "failed_physical_capacity",
            "iterations": 2,
            "forward": True,
            "detection_loss": True,
            "backward": True,
            "optimizer_step": True,
            "last_loss_items": last_loss_items,
            "duration_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": peak,
            "peak_allocated_gib": peak / 1024**3,
            "gpu_total_bytes": int(total_memory),
            "gpu_free_bytes_before": int(free_before),
            "physical_vram_utilization": peak / total_memory if total_memory else None,
            "required_physical_vram_headroom_fraction": 0.50,
        }
    except BaseException as error:
        if not is_cuda_oom(error):
            raise
        return {
            "batch": batch_size,
            "status": "failed_oom",
            "error": str(error),
            "duration_seconds": time.perf_counter() - started,
        }
    finally:
        del optimizer, network, probe
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run E7 YOLO11s-P2-LFR using train only and complete val.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    args = parse_args()
    started_at = utc_now()
    started = time.perf_counter()
    assert_output_available(OUTPUT_DIR)
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    sidecars = (PREFLIGHT_SIDECAR, TRANSFER_SIDECAR, PARAMETERS_SIDECAR, INIT_CHECKPOINT, CONSOLE_LOG)
    existing_sidecars = [str(path.resolve()) for path in sidecars if path.exists()]
    if existing_sidecars:
        raise FileExistsError(f"refusing to overwrite existing E7 sidecars: {existing_sidecars}")
    for required in (E4_WEIGHT, E6_CONFIG, MODEL_CONFIG, MODULE_SOURCE, DATASET_YAML):
        if not required.is_file():
            raise FileNotFoundError(required.resolve())
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} is unavailable")

    with CONSOLE_LOG.open("x", encoding="utf-8", buffering=1) as log_handle:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = TeeStream(original_stdout, log_handle)  # type: ignore[assignment]
        sys.stderr = TeeStream(original_stderr, log_handle)  # type: ignore[assignment]
        try:
            print(f"E7_STARTED_AT_UTC={started_at}", flush=True)
            data_contract = validate_dataset_contract()
            register_lfr_module()
            source = YOLO(str(E4_WEIGHT.resolve()), task="detect")
            e6 = YOLO(str(E6_CONFIG.resolve()), task="detect")
            target = YOLO(str(MODEL_CONFIG.resolve()), task="detect")
            source_strides = [float(value) for value in source.model.stride.detach().cpu().tolist()]
            e6_strides = [float(value) for value in e6.model.stride.detach().cpu().tolist()]
            target_strides = [float(value) for value in target.model.stride.detach().cpu().tolist()]
            if source_strides != [8.0, 16.0, 32.0]:
                raise RuntimeError(f"E4 stride contract failed: {source_strides}")
            if e6_strides != [4.0, 8.0, 16.0, 32.0] or target_strides != e6_strides:
                raise RuntimeError(f"E7 stride contract failed: E6={e6_strides}, E7={target_strides}")
            if dict(source.names) != EXPECTED_CLASSES:
                raise RuntimeError(f"E4 class mapping changed: {source.names}")
            if [type(layer) for layer in target.model.model[:26]] != [type(layer) for layer in e6.model.model[:26]]:
                raise RuntimeError("E7 changed an E6 layer before the P2 refinement insertion")
            if not isinstance(target.model.model[E7_LFR_INDEX], LiteFeatureRefinement):
                raise RuntimeError("E7 LFR layer missing at index 26")
            detect_sources = list(target.model.model[E7_DETECT_INDEX].f)
            if detect_sources != [26, 16, 19, 22]:
                raise RuntimeError(f"E7 Detect sources changed: {detect_sources}")

            initial_target_state = target.model.state_dict()
            transferred_state, transfer_report = transfer_e4_state(source.model.state_dict(), initial_target_state)
            unexpected_random = [
                name
                for name in transfer_report["random_initialized_tensors"]
                if not name.startswith("model.25.")
                and not name.startswith("model.26.")
                and not name.startswith("model.27.cv2.0.")
                and not name.startswith("model.27.cv3.0.")
            ]
            if unexpected_random:
                raise RuntimeError(f"non-P2/LFR tensors remain randomly initialized: {unexpected_random}")
            target.model.load_state_dict(transferred_state, strict=True)
            target.model.names = dict(EXPECTED_CLASSES)
            loaded_state = target.model.state_dict()
            copied_targets = set(transfer_report["exact_tensors"])
            copied_targets.update(item["target"] for item in transfer_report["remapped_tensors"])
            if any(not torch.equal(loaded_state[name].cpu(), transferred_state[name].cpu()) for name in copied_targets):
                raise RuntimeError("post-load transfer integrity check failed")
            random_names = transfer_report["random_initialized_tensors"]
            if any(not torch.equal(loaded_state[name].cpu(), initial_target_state[name].cpu()) for name in random_names):
                raise RuntimeError("P2/LFR random-initialization integrity check failed")

            device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
            lfr_shapes: dict[str, list[int]] = {}

            def shape_hook(_module: object, inputs: tuple[object, ...], output: object) -> None:
                lfr_shapes["input"] = list(inputs[0].shape)  # type: ignore[union-attr]
                lfr_shapes["output"] = list(output.shape)  # type: ignore[union-attr]

            hook = target.model.model[E7_LFR_INDEX].register_forward_hook(shape_hook)
            target.model.to(device).eval()
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                forward_output = target.model(torch.zeros((1, 3, 960, 960), device=device))
            hook.remove()
            leaves = list(tensor_leaves(forward_output))
            if not leaves or not all(torch.isfinite(item).all().item() for item in leaves):
                raise RuntimeError("E7 960 forward returned no tensors or non-finite values")
            if lfr_shapes.get("input") != lfr_shapes.get("output") or lfr_shapes.get("input") != [1, 128, 240, 240]:
                raise RuntimeError(f"LFR input/output shape contract failed: {lfr_shapes}")
            eval_shapes = [list(item.shape) for item in leaves]
            target.model.cpu().eval()
            del forward_output, leaves
            if device.type == "cuda":
                torch.cuda.empty_cache()

            parameters = sum(parameter.numel() for parameter in target.model.parameters())
            trainable_parameters = sum(parameter.numel() for parameter in target.model.parameters() if parameter.requires_grad)
            lfr_parameters = sum(parameter.numel() for parameter in target.model.model[E7_LFR_INDEX].parameters())
            gflops_960 = float(get_flops(target.model, imgsz=960))

            probe_reports: list[dict[str, object]] = []
            cpu_initial_state = {name: tensor.detach().cpu().clone() for name, tensor in loaded_state.items()}
            for batch_size in PROBE_BATCHES:
                print(f"E7_BATCH_PROBE_START batch={batch_size}", flush=True)
                probe_report = run_batch_probe(
                    batch_size=batch_size, initial_state=cpu_initial_state, device=device
                )
                probe_reports.append(probe_report)
                print(json.dumps(probe_report, ensure_ascii=False), flush=True)
            passed_batches = [int(row["batch"]) for row in probe_reports if row["status"] == "passed"]
            if 2 not in passed_batches:
                raise RuntimeError(f"required batch=2 feasibility gate failed: {probe_reports}")
            selected_batch = max(passed_batches)

            transfer_report.update(
                {
                    "source_weight": str(E4_WEIGHT.resolve()),
                    "source_weight_sha256": sha256(E4_WEIGHT),
                    "source_detect_strides": source_strides,
                    "target_detect_strides": target_strides,
                    "model_config": str(MODEL_CONFIG.resolve()),
                    "random_initialization_scope": "E6 P2 fusion layer 25, E7 LFR layer 26, and Detect P2 branch 0",
                }
            )
            write_json(TRANSFER_SIDECAR, transfer_report)
            preflight = {
                "status": "passed",
                "checked_at_utc": utc_now(),
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                },
                "architecture": {
                    "e4_detect_strides": source_strides,
                    "e6_detect_strides": e6_strides,
                    "e7_detect_strides": target_strides,
                    "e7_detect_sources": detect_sources,
                    "p3_p4_p5_and_prior_layers_match_e6": True,
                    "lfr_index": E7_LFR_INDEX,
                    "lfr_input_shape": lfr_shapes["input"],
                    "lfr_output_shape": lfr_shapes["output"],
                    "parameters": parameters,
                    "trainable_parameters": trainable_parameters,
                    "lfr_parameters": lfr_parameters,
                    "gflops_at_960": gflops_960,
                },
                "weight_transfer": {
                    key: transfer_report[key]
                    for key in (
                        "transferred_tensor_count",
                        "transferred_elements",
                        "random_initialized_tensor_count",
                        "random_initialized_elements",
                    )
                },
                "forward_test": {
                    "status": "passed",
                    "input_shape": [1, 3, 960, 960],
                    "output_tensor_shapes": eval_shapes,
                    "finite": True,
                },
                "batch_feasibility": {
                    "probe_order": list(PROBE_BATCHES),
                    "reports": probe_reports,
                    "selected_maximum_stable_batch": selected_batch,
                    "nbs": 64,
                },
                "data_contract": data_contract,
            }
            write_json(PREFLIGHT_SIDECAR, preflight)
            print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)

            target.save(str(INIT_CHECKPOINT.resolve()))
            del source, e6, target
            if device.type == "cuda":
                torch.cuda.empty_cache()
            register_lfr_module()
            training_model = YOLO(str(INIT_CHECKPOINT.resolve()), task="detect")
            if [float(value) for value in training_model.model.stride.detach().cpu().tolist()] != target_strides:
                raise RuntimeError("saved E7 initialization checkpoint lost four detection strides")
            if dict(training_model.names) != EXPECTED_CLASSES:
                raise RuntimeError("saved E7 initialization checkpoint lost class mapping")
            if not isinstance(training_model.model.model[E7_LFR_INDEX], LiteFeatureRefinement):
                raise RuntimeError("saved E7 initialization checkpoint lost project-local LFR")

            training_kwargs = build_training_kwargs(
                data_yaml=DATASET_YAML,
                output_dir=OUTPUT_DIR,
                device=args.device,
                workers=args.workers,
                batch=selected_batch,
            )
            write_json(
                PARAMETERS_SIDECAR,
                {
                    "experiment_id": "E7",
                    "training_parameters": training_kwargs,
                    "requested_initial_batch": 8,
                    "probe_order": list(PROBE_BATCHES),
                    "selected_actual_batch": selected_batch,
                    "nbs": 64,
                    "source_experiment_for_hyperparameters": "E6",
                    "test_used": False,
                },
            )
            print("E7_PRECHECK_PASSED_STARTING_TRAINING", flush=True)
            print(json.dumps(training_kwargs, ensure_ascii=False, indent=2), flush=True)
            training_model.train(**training_kwargs)

            best_pt = OUTPUT_DIR / "weights" / "best.pt"
            last_pt = OUTPUT_DIR / "weights" / "last.pt"
            results_csv = OUTPUT_DIR / "results.csv"
            for required in (best_pt, last_pt, results_csv):
                if not required.is_file():
                    raise FileNotFoundError(f"training output missing: {required.resolve()}")
            for source_path, output_name in (
                (PREFLIGHT_SIDECAR, "e7_preflight_report.json"),
                (TRANSFER_SIDECAR, "e4_to_e7_weight_transfer_report.json"),
                (PARAMETERS_SIDECAR, "e7_training_parameters.json"),
                (MODEL_CONFIG, "yolo11s-p2-lfr.yaml"),
                (MODULE_SOURCE, "e7_lfr_module.py"),
            ):
                shutil.copy2(source_path, OUTPUT_DIR / output_name)
            training_report = {
                "status": "passed",
                "experiment_id": "E7",
                "started_at_utc": started_at,
                "ended_at_utc": utc_now(),
                "duration_seconds": time.perf_counter() - started,
                "training_parameters": training_kwargs,
                "requested_initial_batch": 8,
                "actual_batch": selected_batch,
                "nbs": 64,
                "initialization_checkpoint": str(INIT_CHECKPOINT.resolve()),
                "initialization_checkpoint_sha256": sha256(INIT_CHECKPOINT),
                "e4_source_weight": str(E4_WEIGHT.resolve()),
                "e4_source_weight_sha256": transfer_report["source_weight_sha256"],
                "model_parameters": parameters,
                "model_gflops_at_960": gflops_960,
                "outputs": {
                    "run_dir": str(OUTPUT_DIR.resolve()),
                    "best_pt": str(best_pt.resolve()),
                    "best_pt_sha256": sha256(best_pt),
                    "last_pt": str(last_pt.resolve()),
                    "last_pt_sha256": sha256(last_pt),
                    "results_csv": str(results_csv.resolve()),
                    "console_log": str((OUTPUT_DIR / "e7_training.log").resolve()),
                },
                "data_contract": data_contract,
                "test_used": False,
            }
            write_json(OUTPUT_DIR / "e7_training_report.json", training_report)
            print(json.dumps(training_report, ensure_ascii=False, indent=2), flush=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    if OUTPUT_DIR.is_dir():
        shutil.copy2(CONSOLE_LOG, OUTPUT_DIR / "e7_training.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
