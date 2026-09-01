#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import scripts.train.train_e7_lfr as e7_runner  # noqa: E402
from helmet_safety.training.e7_lfr import LiteFeatureRefinement  # noqa: E402
from helmet_safety.training.e8_no_eca import (  # noqa: E402
    E8_DETECT_INDEX,
    E8_REFINEMENT_INDEX,
    NoECAFeatureRefinement,
    build_training_kwargs,
    register_no_eca_module,
    transfer_e4_state,
)


E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E6_CONFIG = PROJECT_ROOT / "configs" / "yolo11s-p2.yaml"
E7_CONFIG = PROJECT_ROOT / "configs" / "yolo11s-p2-lfr.yaml"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "yolo11s-p2-lfr-no-eca.yaml"
MODULE_SOURCE = PROJECT_ROOT / "src" / "helmet_safety" / "training" / "e8_no_eca.py"
DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
E7_PARAMETERS = PROJECT_ROOT / "artifacts" / "e7" / "e7_yolo11s_p2_lfr_001" / "e7_training_parameters.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "e8" / "e8_yolo11s_p2_lfr_no_eca_001"
SIDECAR_PREFIX = OUTPUT_DIR.parent / OUTPUT_DIR.name
PREFLIGHT_SIDECAR = Path(f"{SIDECAR_PREFIX}_preflight.json")
TRANSFER_SIDECAR = Path(f"{SIDECAR_PREFIX}_weight_transfer.json")
PARAMETERS_SIDECAR = Path(f"{SIDECAR_PREFIX}_training_parameters.json")
INIT_CHECKPOINT = Path(f"{SIDECAR_PREFIX}_init.pt")
CONSOLE_LOG = Path(f"{SIDECAR_PREFIX}_training.log")
EXPECTED_CLASSES = {0: "helmet", 1: "no_helmet"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E8 ECA ablation on original train and complete val.")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def assert_e7_e8_only_attention_diff(e7: object, e8: object) -> dict[str, object]:
    e7_yaml = yaml.safe_load(E7_CONFIG.read_text(encoding="utf-8"))
    e8_yaml = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    e7_layer_name = e7_yaml["head"][15][2]
    e7_yaml["head"][15][2] = e8_yaml["head"][15][2]
    if e7_yaml != e8_yaml or e7_layer_name != "LiteFeatureRefinement":
        raise RuntimeError("E8 YAML differs from E7 beyond replacing the refinement layer")

    e7_layers = e7.model.model  # type: ignore[attr-defined]
    e8_layers = e8.model.model  # type: ignore[attr-defined]
    if len(e7_layers) != len(e8_layers):
        raise RuntimeError("E8 changed the E7 layer count")
    for index, (left, right) in enumerate(zip(e7_layers, e8_layers, strict=True)):
        if index == E8_REFINEMENT_INDEX:
            continue
        if type(left) is not type(right) or left.f != right.f:
            raise RuntimeError(f"E8 changed E7 layer {index}")
    if not isinstance(e7_layers[E8_REFINEMENT_INDEX], LiteFeatureRefinement):
        raise RuntimeError("E7 refinement layer is unavailable for ablation comparison")
    if not isinstance(e8_layers[E8_REFINEMENT_INDEX], NoECAFeatureRefinement):
        raise RuntimeError("E8 no-ECA refinement layer is missing")
    e7_children = [name for name, _ in e7_layers[E8_REFINEMENT_INDEX].named_children()]
    e8_children = [name for name, _ in e8_layers[E8_REFINEMENT_INDEX].named_children()]
    if e8_children != [name for name in e7_children if name != "channel_attention"]:
        raise RuntimeError(f"E8 refinement changed beyond ECA removal: E7={e7_children}, E8={e8_children}")

    e7_state = e7.model.state_dict()  # type: ignore[attr-defined]
    e8_state = e8.model.state_dict()  # type: ignore[attr-defined]
    removed = sorted(set(e7_state) - set(e8_state))
    added = sorted(set(e8_state) - set(e7_state))
    expected_removed = [f"model.{E8_REFINEMENT_INDEX}.channel_attention.conv.weight"]
    if removed != expected_removed or added:
        raise RuntimeError(f"unexpected E7/E8 state difference: removed={removed}, added={added}")
    shape_changes = [name for name in e8_state if e8_state[name].shape != e7_state[name].shape]
    if shape_changes:
        raise RuntimeError(f"E8 changed non-ECA tensor shapes: {shape_changes}")
    return {
        "passed": True,
        "yaml_only_change": {"from": "LiteFeatureRefinement", "to": "NoECAFeatureRefinement"},
        "removed_modules": ["channel_attention (AdaptiveAvgPool2d + Conv1d + Sigmoid behavior)"],
        "removed_state_tensors": removed,
        "added_state_tensors": added,
        "all_other_layer_types_sources_and_tensor_shapes_equal": True,
        "preserved_operations": ["DWConv3x3", "BatchNorm2d", "SiLU", "PWConv1x1", "BatchNorm2d", "residual_add"],
    }


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    args = parse_args()
    started_at = utc_now()
    started = time.perf_counter()
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing E8 output: {OUTPUT_DIR.resolve()}")
    sidecars = (PREFLIGHT_SIDECAR, TRANSFER_SIDECAR, PARAMETERS_SIDECAR, INIT_CHECKPOINT, CONSOLE_LOG)
    existing = [str(path.resolve()) for path in sidecars if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing E8 sidecars: {existing}")
    for required in (E4_WEIGHT, E6_CONFIG, E7_CONFIG, MODEL_CONFIG, MODULE_SOURCE, DATASET_YAML, E7_PARAMETERS):
        if not required.is_file():
            raise FileNotFoundError(required.resolve())
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} is unavailable")
    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)

    with CONSOLE_LOG.open("x", encoding="utf-8", buffering=1) as log_handle:
        original_stdout, original_stderr = sys.stdout, sys.stderr
        sys.stdout = e7_runner.TeeStream(original_stdout, log_handle)  # type: ignore[assignment]
        sys.stderr = e7_runner.TeeStream(original_stderr, log_handle)  # type: ignore[assignment]
        try:
            print(f"E8_STARTED_AT_UTC={started_at}", flush=True)
            data_contract = e7_runner.validate_dataset_contract()
            register_no_eca_module()
            e7_runner.register_lfr_module()
            source = YOLO(str(E4_WEIGHT.resolve()), task="detect")
            e6 = YOLO(str(E6_CONFIG.resolve()), task="detect")
            e7 = YOLO(str(E7_CONFIG.resolve()), task="detect")
            target = YOLO(str(MODEL_CONFIG.resolve()), task="detect")

            source_strides = [float(value) for value in source.model.stride.cpu().tolist()]
            e6_strides = [float(value) for value in e6.model.stride.cpu().tolist()]
            target_strides = [float(value) for value in target.model.stride.cpu().tolist()]
            if source_strides != [8.0, 16.0, 32.0]:
                raise RuntimeError(f"E4 stride contract failed: {source_strides}")
            if target_strides != [4.0, 8.0, 16.0, 32.0] or target_strides != e6_strides:
                raise RuntimeError(f"E8 stride contract failed: E6={e6_strides}, E8={target_strides}")
            if dict(source.names) != EXPECTED_CLASSES:
                raise RuntimeError(f"E4 class mapping changed: {source.names}")
            architecture_ablation = assert_e7_e8_only_attention_diff(e7, target)
            detect_sources = list(target.model.model[E8_DETECT_INDEX].f)
            if detect_sources != [26, 16, 19, 22]:
                raise RuntimeError(f"E8 Detect sources changed: {detect_sources}")

            initial_target_state = target.model.state_dict()
            transferred_state, transfer_report = transfer_e4_state(source.model.state_dict(), initial_target_state)
            unexpected_random = [
                name for name in transfer_report["random_initialized_tensors"]
                if not name.startswith("model.25.")
                and not name.startswith("model.26.")
                and not name.startswith("model.27.cv2.0.")
                and not name.startswith("model.27.cv3.0.")
            ]
            if unexpected_random:
                raise RuntimeError(f"non-P2/E8 tensors remain randomly initialized: {unexpected_random}")
            target.model.load_state_dict(transferred_state, strict=True)
            target.model.names = dict(EXPECTED_CLASSES)
            loaded_state = target.model.state_dict()
            copied = set(transfer_report["exact_tensors"])
            copied.update(item["target"] for item in transfer_report["remapped_tensors"])
            if any(not torch.equal(loaded_state[name].cpu(), transferred_state[name].cpu()) for name in copied):
                raise RuntimeError("post-load E4 transfer integrity check failed")
            random_names = transfer_report["random_initialized_tensors"]
            if any(not torch.equal(loaded_state[name].cpu(), initial_target_state[name].cpu()) for name in random_names):
                raise RuntimeError("E8 random-initialization integrity check failed")

            device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
            refinement_shapes: dict[str, list[int]] = {}
            def shape_hook(_module: object, inputs: tuple[object, ...], output: object) -> None:
                refinement_shapes["input"] = list(inputs[0].shape)  # type: ignore[union-attr]
                refinement_shapes["output"] = list(output.shape)  # type: ignore[union-attr]

            hook = target.model.model[E8_REFINEMENT_INDEX].register_forward_hook(shape_hook)
            target.model.to(device).eval()
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                forward_output = target.model(torch.zeros((1, 3, 960, 960), device=device))
            hook.remove()
            leaves = list(e7_runner.tensor_leaves(forward_output))
            if not leaves or not all(torch.isfinite(item).all().item() for item in leaves):
                raise RuntimeError("E8 imgsz=960 forward returned no tensors or non-finite values")
            if refinement_shapes.get("input") != [1, 128, 240, 240] or refinement_shapes.get("output") != refinement_shapes.get("input"):
                raise RuntimeError(f"E8 refinement shape contract failed: {refinement_shapes}")
            eval_shapes = [list(item.shape) for item in leaves]
            target.model.cpu().eval()
            del forward_output, leaves
            if device.type == "cuda":
                torch.cuda.empty_cache()

            parameters = sum(parameter.numel() for parameter in target.model.parameters())
            trainable_parameters = sum(parameter.numel() for parameter in target.model.parameters() if parameter.requires_grad)
            refinement_parameters = sum(parameter.numel() for parameter in target.model.model[E8_REFINEMENT_INDEX].parameters())
            gflops_960 = float(get_flops(target.model, imgsz=960))

            e7_runner.MODEL_CONFIG = MODEL_CONFIG
            e7_runner.register_lfr_module = register_no_eca_module
            cpu_initial_state = {name: tensor.detach().cpu().clone() for name, tensor in loaded_state.items()}
            print("E8_BATCH2_VRAM_PROBE_START", flush=True)
            memory_report = e7_runner.run_batch_probe(batch_size=2, initial_state=cpu_initial_state, device=device)
            print(json.dumps(memory_report, ensure_ascii=False), flush=True)
            if memory_report["status"] != "passed":
                raise RuntimeError(f"required batch=2 VRAM feasibility gate failed: {memory_report}")

            training_kwargs = build_training_kwargs(
                data_yaml=DATASET_YAML, output_dir=OUTPUT_DIR, device=args.device, workers=args.workers, batch=2
            )
            e7_parameters = json.loads(E7_PARAMETERS.read_text(encoding="utf-8"))["training_parameters"]
            allowed_differences = {"project", "name"}
            parameter_differences = {
                key: {"e7": e7_parameters.get(key), "e8": training_kwargs.get(key)}
                for key in set(e7_parameters) | set(training_kwargs)
                if key not in allowed_differences and e7_parameters.get(key) != training_kwargs.get(key)
            }
            if parameter_differences:
                raise RuntimeError(f"E8 training parameters differ from E7: {parameter_differences}")

            transfer_report.update({
                "source_weight": str(E4_WEIGHT.resolve()),
                "source_weight_sha256": e7_runner.sha256(E4_WEIGHT),
                "source_detect_strides": source_strides,
                "target_detect_strides": target_strides,
                "model_config": str(MODEL_CONFIG.resolve()),
                "successful_transfer": {
                    "tensor_count": transfer_report["transferred_tensor_count"],
                    "elements": transfer_report["transferred_elements"],
                    "exact_tensors": transfer_report["exact_tensors"],
                    "remapped_tensors": transfer_report["remapped_tensors"],
                },
                "random_initialization": {
                    "tensor_count": transfer_report["random_initialized_tensor_count"],
                    "elements": transfer_report["random_initialized_elements"],
                    "tensors": transfer_report["random_initialized_tensors"],
                    "scope": "E6 P2 fusion layer 25, E8 no-ECA refinement layer 26, and Detect P2 branch 0",
                },
            })
            e7_runner.write_json(TRANSFER_SIDECAR, transfer_report)
            preflight = {
                "status": "passed",
                "formal_training_started": False,
                "checked_at_utc": utc_now(),
                "environment": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "ultralytics": ultralytics.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                },
                "architecture": {
                    "e8_detect_strides": target_strides,
                    "e8_detect_sources": detect_sources,
                    "e7_to_e8_ablation": architecture_ablation,
                    "refinement_input_shape": refinement_shapes["input"],
                    "refinement_output_shape": refinement_shapes["output"],
                    "parameters": parameters,
                    "trainable_parameters": trainable_parameters,
                    "refinement_parameters": refinement_parameters,
                    "gflops_at_960": gflops_960,
                },
                "weight_transfer": {
                    "status": "passed",
                    "report": str(TRANSFER_SIDECAR.resolve()),
                    "transferred_tensor_count": transfer_report["transferred_tensor_count"],
                    "random_initialized_tensor_count": transfer_report["random_initialized_tensor_count"],
                },
                "forward_test": {"status": "passed", "input_shape": [1, 3, 960, 960], "output_tensor_shapes": eval_shapes, "finite": True},
                "vram_check": memory_report,
                "training_parameter_parity_with_e7": {"status": "passed", "allowed_output_path_keys": sorted(allowed_differences)},
                "data_contract": data_contract,
            }
            e7_runner.write_json(PREFLIGHT_SIDECAR, preflight)
            e7_runner.write_json(PARAMETERS_SIDECAR, {
                "experiment_id": "E8",
                "training_parameters": training_kwargs,
                "actual_batch": 2,
                "nbs": 64,
                "source_experiment_for_hyperparameters_and_augmentation": "E7",
                "test_used": False,
            })
            print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)

            target.save(str(INIT_CHECKPOINT.resolve()))
            del source, e6, e7, target
            if device.type == "cuda":
                torch.cuda.empty_cache()
            register_no_eca_module()
            training_model = YOLO(str(INIT_CHECKPOINT.resolve()), task="detect")
            if [float(value) for value in training_model.model.stride.cpu().tolist()] != target_strides:
                raise RuntimeError("saved E8 initialization checkpoint lost four detection strides")
            if not isinstance(training_model.model.model[E8_REFINEMENT_INDEX], NoECAFeatureRefinement):
                raise RuntimeError("saved E8 checkpoint lost the project-local refinement")

            print("E8_ALL_PREFLIGHT_GATES_PASSED_STARTING_FORMAL_TRAINING", flush=True)
            print(json.dumps(training_kwargs, ensure_ascii=False, indent=2), flush=True)
            training_model.train(**training_kwargs)

            required_outputs = [OUTPUT_DIR / "weights" / "best.pt", OUTPUT_DIR / "weights" / "last.pt", OUTPUT_DIR / "results.csv"]
            for required in required_outputs:
                if not required.is_file():
                    raise FileNotFoundError(f"training output missing: {required.resolve()}")
            for source_path, output_name in (
                (PREFLIGHT_SIDECAR, "e8_preflight_report.json"),
                (TRANSFER_SIDECAR, "e4_to_e8_weight_transfer_report.json"),
                (PARAMETERS_SIDECAR, "e8_training_parameters.json"),
                (MODEL_CONFIG, MODEL_CONFIG.name),
                (MODULE_SOURCE, "e8_no_eca_module.py"),
            ):
                shutil.copy2(source_path, OUTPUT_DIR / output_name)
            training_report = {
                "status": "passed", "experiment_id": "E8", "started_at_utc": started_at,
                "ended_at_utc": utc_now(), "duration_seconds": time.perf_counter() - started,
                "training_parameters": training_kwargs, "actual_batch": 2, "nbs": 64,
                "initialization_checkpoint": str(INIT_CHECKPOINT.resolve()),
                "initialization_checkpoint_sha256": e7_runner.sha256(INIT_CHECKPOINT),
                "e4_source_weight": str(E4_WEIGHT.resolve()),
                "e4_source_weight_sha256": transfer_report["source_weight_sha256"],
                "model_parameters": parameters, "model_gflops_at_960": gflops_960,
                "outputs": {
                    "run_dir": str(OUTPUT_DIR.resolve()),
                    "best_pt": str(required_outputs[0].resolve()), "best_pt_sha256": e7_runner.sha256(required_outputs[0]),
                    "last_pt": str(required_outputs[1].resolve()), "last_pt_sha256": e7_runner.sha256(required_outputs[1]),
                    "results_csv": str(required_outputs[2].resolve()),
                },
                "data_contract": data_contract, "test_used": False,
            }
            e7_runner.write_json(OUTPUT_DIR / "e8_training_report.json", training_report)
            print(json.dumps(training_report, ensure_ascii=False, indent=2), flush=True)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    if OUTPUT_DIR.is_dir():
        shutil.copy2(CONSOLE_LOG, OUTPUT_DIR / "e8_training.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
