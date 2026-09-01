from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

import torch
from torch import nn


E4_DETECT_INDEX = 23
E7_LFR_INDEX = 26
E7_DETECT_INDEX = 27
PHYSICAL_VRAM_UTILIZATION_LIMIT = 0.50


class EfficientChannelAttention(nn.Module):
    """ECA-style channel attention with one parameter-light 1D convolution."""

    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1).transpose(-1, -2)
        weights = self.gate(self.conv(weights)).transpose(-1, -2).unsqueeze(-1)
        return x * weights


class LiteFeatureRefinement(nn.Module):
    """Shape-preserving P2 refinement using DW/PW convolutions, ECA and a residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(channels)
        self.activation = nn.SiLU(inplace=True)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(channels)
        self.channel_attention = EfficientChannelAttention(kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        refined = self.activation(self.depthwise_bn(self.depthwise(x)))
        refined = self.pointwise_bn(self.pointwise(refined))
        refined = self.channel_attention(refined)
        return x + refined


def register_lfr_module() -> None:
    """Expose the project-local layer to Ultralytics' YAML parser without editing the package."""

    import ultralytics.nn.tasks as tasks

    tasks.__dict__[LiteFeatureRefinement.__name__] = LiteFeatureRefinement


def configure_probe_hyperparameters(network: nn.Module) -> None:
    """Attach E6-compatible attribute-style loss settings to a YAML-built model."""

    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    network.args = get_cfg(
        DEFAULT_CFG,
        overrides={"box": 7.5, "cls": 0.5, "dfl": 1.5},
    )


def fits_physical_vram(*, peak_allocated_bytes: int, total_bytes: int) -> bool:
    """Reserve 50% for dense real batches beyond the one-target synthetic probe."""

    if peak_allocated_bytes < 0 or total_bytes <= 0:
        raise ValueError("CUDA memory byte counts must be positive")
    return peak_allocated_bytes <= total_bytes * PHYSICAL_VRAM_UTILIZATION_LIMIT


def assert_output_available(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing E7 output: {output_dir.resolve()}")


def build_resume_plan(
    *,
    checkpoint: Mapping[str, object],
    checkpoint_path: Path,
    completed_epochs: int,
    best_epoch: int,
    device: str,
    workers: int,
) -> dict[str, object]:
    """Validate an interrupted E7 checkpoint and describe an exact-state resume."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path.resolve())
    checkpoint_epoch = int(checkpoint.get("epoch", -1)) + 1
    if checkpoint_epoch != completed_epochs:
        raise RuntimeError(
            f"checkpoint/CSV epoch mismatch: checkpoint={checkpoint_epoch}, CSV={completed_epochs}"
        )
    if checkpoint.get("optimizer") is None or checkpoint.get("scaler") is None:
        raise RuntimeError("checkpoint is not resumable: optimizer or scaler state is missing")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, Mapping):
        raise RuntimeError("checkpoint is not resumable: train_args are missing")
    expected = {
        "epochs": 50,
        "patience": 15,
        "batch": 2,
        "imgsz": 960,
        "nbs": 64,
        "seed": 42,
        "deterministic": True,
    }
    changed = {
        key: {"expected": value, "actual": train_args.get(key)}
        for key, value in expected.items()
        if train_args.get(key) != value
    }
    if changed:
        raise RuntimeError(f"checkpoint training contract changed: {changed}")
    total_epochs = int(train_args["epochs"])
    if not 0 < completed_epochs < total_epochs:
        raise RuntimeError(
            f"checkpoint has no remaining epochs: completed={completed_epochs}, total={total_epochs}"
        )
    if not 0 < best_epoch <= completed_epochs:
        raise RuntimeError(f"invalid best epoch {best_epoch} for {completed_epochs} completed epochs")
    best_fitness = float(checkpoint.get("best_fitness", 0.0))
    if best_fitness <= 0.0:
        raise RuntimeError(f"invalid checkpoint best_fitness: {best_fitness}")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "completed_epochs": completed_epochs,
        "next_epoch": completed_epochs + 1,
        "total_epochs": total_epochs,
        "remaining_epochs": total_epochs - completed_epochs,
        "best_epoch": best_epoch,
        "best_fitness": best_fitness,
        "train_kwargs": {"resume": True, "device": device, "workers": workers},
    }


def restore_early_stopping(trainer: object, *, best_epoch: int, best_fitness: float) -> None:
    """Restore patience history that Ultralytics does not serialize in its checkpoint."""

    stopper = trainer.stopper  # type: ignore[attr-defined]
    stopper.best_epoch = best_epoch
    stopper.best_fitness = best_fitness
    stopper.possible_stop = False


def build_training_kwargs(
    *, data_yaml: Path, output_dir: Path, device: str, workers: int, batch: int
) -> dict[str, object]:
    """Return the E6 training settings with only the selected feasible batch changed."""

    return {
        "data": str(data_yaml.resolve()),
        "epochs": 50,
        "patience": 15,
        "batch": batch,
        "imgsz": 960,
        "save": True,
        "save_period": -1,
        "cache": False,
        "device": device,
        "workers": workers,
        "project": str(output_dir.parent.resolve()),
        "name": output_dir.name,
        "exist_ok": False,
        "pretrained": True,
        "optimizer": "auto",
        "verbose": True,
        "seed": 42,
        "deterministic": True,
        "single_cls": False,
        "rect": False,
        "cos_lr": False,
        "close_mosaic": 10,
        "resume": False,
        "amp": True,
        "fraction": 1.0,
        "profile": False,
        "freeze": None,
        "multi_scale": 0.0,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        "val": True,
        "split": "val",
        "save_json": False,
        "iou": 0.7,
        "max_det": 300,
        "plots": True,
        "augment": False,
        "agnostic_nms": False,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "nbs": 64,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
    }


def _source_key_for_target(target_key: str) -> str | None:
    target_prefix = f"model.{E7_DETECT_INDEX}."
    if not target_key.startswith(target_prefix):
        return None
    suffix = target_key[len(target_prefix) :]
    if suffix.startswith("dfl."):
        return f"model.{E4_DETECT_INDEX}.{suffix}"
    match = re.match(r"(cv[23])\.(\d+)\.(.+)", suffix)
    if not match:
        return None
    branch = int(match.group(2))
    if branch == 0:
        return None
    return f"model.{E4_DETECT_INDEX}.{match.group(1)}.{branch - 1}.{match.group(3)}"


def transfer_e4_state(
    source_state: Mapping[str, torch.Tensor], target_state: Mapping[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Copy exact compatible tensors and remap E4 Detect P3-P5 into E7 branches 1-3."""

    transferred = {name: tensor.detach().clone() for name, tensor in target_state.items()}
    exact: list[str] = []
    remapped: list[dict[str, str]] = []
    copied_targets: set[str] = set()

    for target_name, target_tensor in target_state.items():
        source_tensor = source_state.get(target_name)
        if source_tensor is not None and source_tensor.shape == target_tensor.shape:
            transferred[target_name] = source_tensor.detach().clone()
            exact.append(target_name)
            copied_targets.add(target_name)

    for target_name, target_tensor in target_state.items():
        source_name = _source_key_for_target(target_name)
        if source_name is None:
            continue
        source_tensor = source_state.get(source_name)
        if source_tensor is None or source_tensor.shape != target_tensor.shape:
            continue
        transferred[target_name] = source_tensor.detach().clone()
        remapped.append({"source": source_name, "target": target_name})
        copied_targets.add(target_name)

    random_initialized = sorted(set(target_state) - copied_targets)
    copied_elements = sum(int(target_state[name].numel()) for name in copied_targets)
    random_elements = sum(int(target_state[name].numel()) for name in random_initialized)
    return transferred, {
        "exact_tensor_count": len(exact),
        "exact_tensors": sorted(exact),
        "remapped_tensor_count": len(remapped),
        "remapped_tensors": remapped,
        "transferred_tensor_count": len(copied_targets),
        "transferred_elements": copied_elements,
        "random_initialized_tensor_count": len(random_initialized),
        "random_initialized_elements": random_elements,
        "random_initialized_tensors": random_initialized,
        "target_state_tensor_count": len(target_state),
        "target_state_elements": copied_elements + random_elements,
    }


def build_e7_comparison(
    e6: Mapping[str, object], e7: Mapping[str, object]
) -> dict[str, object]:
    """Build explicit E7-vs-E6 judgments from identically evaluated measurements."""

    def nested(payload: Mapping[str, object], *keys: str) -> float:
        current: object = payload
        for key in keys:
            current = current[key]  # type: ignore[index]
        return float(current)

    def relative(delta: float, baseline: float) -> float:
        return 100.0 * delta / baseline if baseline else (0.0 if delta == 0 else float("inf"))

    tiny_tp_delta = int(nested(e7, "tiny", "tp") - nested(e6, "tiny", "tp"))
    tiny_fn_delta = int(nested(e7, "tiny", "fn") - nested(e6, "tiny", "fn"))
    tiny_recall_pp = 100.0 * (nested(e7, "tiny", "recall") - nested(e6, "tiny", "recall"))
    map_pp = 100.0 * (
        nested(e7, "standard_val", "overall", "map50_95")
        - nested(e6, "standard_val", "overall", "map50_95")
    )
    f1_pp = 100.0 * (
        nested(e7, "standard_val", "overall", "f1")
        - nested(e6, "standard_val", "overall", "f1")
    )
    fp_delta = int(
        nested(e7, "fixed_threshold", "overall", "fp")
        - nested(e6, "fixed_threshold", "overall", "fp")
    )
    confusion_delta = int(nested(e7, "class_confusions") - nested(e6, "class_confusions"))
    parameter_delta = int(nested(e7, "parameters") - nested(e6, "parameters"))
    gflops_delta = nested(e7, "gflops_at_960") - nested(e6, "gflops_at_960")
    fp_relative = relative(fp_delta, nested(e6, "fixed_threshold", "overall", "fp"))
    confusion_relative = relative(confusion_delta, nested(e6, "class_confusions"))
    parameter_relative = relative(parameter_delta, nested(e6, "parameters"))
    gflops_relative = relative(gflops_delta, nested(e6, "gflops_at_960"))

    tiny_reduced = tiny_fn_delta < 0
    overall_improved = map_pp > 0.0 and f1_pp >= 0.0
    fp_increased = fp_delta > 0
    confusion_increased = confusion_delta > 0
    compute_reasonable = parameter_relative <= 2.0 and gflops_relative <= 10.0
    no_material_error_increase = fp_relative <= 10.0 and confusion_relative <= 10.0
    return {
        "rules": {
            "tiny_false_negatives_reduced": "E7 tiny FN is strictly lower than E6",
            "overall_detection_improved": "mAP50-95 rises and standard P/R harmonic F1 does not fall",
            "compute_overhead_reasonable": "parameters rise by <=2% and 960 GFLOPs rise by <=10%",
            "p2_lfr_effective": "tiny FN falls, overall improves, error counts do not rise >10%, and compute is reasonable",
        },
        "deltas": {
            "tiny_tp": tiny_tp_delta,
            "tiny_fn": tiny_fn_delta,
            "tiny_recall_pp": tiny_recall_pp,
            "map50_95_pp": map_pp,
            "standard_f1_pp": f1_pp,
            "false_positives": fp_delta,
            "false_positives_relative_percent": fp_relative,
            "class_confusions": confusion_delta,
            "class_confusions_relative_percent": confusion_relative,
            "parameters": parameter_delta,
            "parameters_relative_percent": parameter_relative,
            "gflops_at_960": gflops_delta,
            "gflops_relative_percent": gflops_relative,
        },
        "tiny_false_negatives_reduced": tiny_reduced,
        "overall_detection_improved": overall_improved,
        "false_positives_increased": fp_increased,
        "class_confusion_increased": confusion_increased,
        "compute_overhead_reasonable": compute_reasonable,
        "p2_lfr_effective": tiny_reduced and overall_improved and no_material_error_increase and compute_reasonable,
    }
