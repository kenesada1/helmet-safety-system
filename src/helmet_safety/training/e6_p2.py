from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

import torch


E4_DETECT_INDEX = 23
E6_DETECT_INDEX = 26


def assert_output_available(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing E6 output: {output_dir.resolve()}")


def build_training_kwargs(
    *, data_yaml: Path, output_dir: Path, device: str, workers: int
) -> dict[str, object]:
    """Return E4-matched training settings with only the P2 architecture changed."""

    return {
        "data": str(data_yaml.resolve()),
        "epochs": 50,
        "patience": 15,
        "batch": 2,
        "imgsz": 960,
        "save": True,
        "save_period": -1,
        "cache": False,
        "device": device,
        "workers": workers,
        "project": str(output_dir.parent.resolve()),
        "name": output_dir.name,
        "exist_ok": False,
        # train() is called on a temporary checkpoint that already contains the
        # explicit E4-to-E6 transfer; True tells Ultralytics to retain it.
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
    target_prefix = f"model.{E6_DETECT_INDEX}."
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
    """Copy all shape-compatible E4 tensors and shift its Detect branches to P3-P5."""

    transferred = {name: tensor.clone() for name, tensor in target_state.items()}
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
    report: dict[str, object] = {
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
    return transferred, report


def build_e6_comparison(
    e4: Mapping[str, object], e6: Mapping[str, object]
) -> dict[str, object]:
    """Compare E6 with E4 using fixed, declared material-change thresholds."""

    def value(payload: Mapping[str, object], *keys: str) -> float:
        current: object = payload
        for key in keys:
            current = current[key]  # type: ignore[index]
        return float(current)

    e4_tiny_recall = value(e4, "tiny", "recall")
    e6_tiny_recall = value(e6, "tiny", "recall")
    e4_fp = value(e4, "fixed_threshold", "overall", "fp")
    e6_fp = value(e6, "fixed_threshold", "overall", "fp")
    precision_pp = 100.0 * (
        value(e6, "fixed_threshold", "overall", "precision")
        - value(e4, "fixed_threshold", "overall", "precision")
    )
    f1_pp = 100.0 * (
        value(e6, "fixed_threshold", "overall", "f1")
        - value(e4, "fixed_threshold", "overall", "f1")
    )
    map_pp = 100.0 * (
        value(e6, "standard_val", "overall", "map50_95")
        - value(e4, "standard_val", "overall", "map50_95")
    )
    fp_relative_percent = 100.0 * (e6_fp - e4_fp) / e4_fp if e4_fp else float("inf")
    return {
        "rules": {
            "tiny_improved": "tiny recall is strictly higher than E4",
            "obvious_false_positive_increase": "full-val FP rises by >10% and precision falls by >1 percentage point",
            "obvious_overall_decline": "mAP50-95 or fixed-threshold F1 falls by >1 percentage point",
        },
        "deltas": {
            "tiny_recall_pp": 100.0 * (e6_tiny_recall - e4_tiny_recall),
            "tiny_tp": int(value(e6, "tiny", "tp") - value(e4, "tiny", "tp")),
            "tiny_fn": int(value(e6, "tiny", "fn") - value(e4, "tiny", "fn")),
            "false_positives": int(e6_fp - e4_fp),
            "false_positives_relative_percent": fp_relative_percent,
            "precision_pp": precision_pp,
            "f1_pp": f1_pp,
            "map50_95_pp": map_pp,
        },
        "tiny_improved": e6_tiny_recall > e4_tiny_recall,
        "obvious_false_positive_increase": fp_relative_percent > 10.0 and precision_pp < -1.0,
        "obvious_overall_decline": map_pp < -1.0 or f1_pp < -1.0,
    }
