#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Mapping, Sequence

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import (  # noqa: E402
    allocate_run_name,
    analyze_training_results,
    load_ground_truth_boxes,
    write_json_report,
)
from helmet_safety.training.e5a import (  # noqa: E402
    audit_training_manifest,
    build_e5a_training_kwargs,
    build_resampled_manifests,
    chinese_detection_count_keys,
    identify_tiny_difficult_samples,
    judge_resampling_effectiveness,
    render_chinese_evaluation_markdown,
    stage_dataset_variant,
    summarize_e5a_evaluation,
    validate_e4_candidate_c_anchor,
)
from helmet_safety.training.analysis_core import validated_streaming_image_source  # noqa: E402


E4_WEIGHT = PROJECT_ROOT / "artifacts" / "training" / "m45_yolo11s_e75_960_001" / "weights" / "best.pt"
E4_TRAINING_REPORT = E4_WEIGHT.parents[2] / "baseline_training_report.json"
E4_VALIDATION_REPORT = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation"
    / "m45_yolo11s_e75_960_full_val_p0_001"
    / "e4_full_val_p0_report.json"
)
DEFAULT_DATASET_YAML = Path(r"D:\datasets\SHWD\processed\dataset.yaml")
EXPECTED_TRAIN_IMAGES = 5457
EXPECTED_TRAIN_BOXES = 86197
EXPECTED_VAL_IMAGES = 607
EXPECTED_VAL_BOXES = 9925
FORMAL_ENTRIES = 6500
EXTRA_ENTRIES = 1043
FIXED_IMGSZ = 960
FIXED_CONFIDENCE = 0.20
FIXED_NMS_IOU = 0.50
FIXED_MATCHING_IOU = 0.50
FIXED_MAX_DET = 300


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E5-A微小困难样本定向重采样分阶段实验入口")
    parser.add_argument(
        "phase",
        choices=("prepare", "smoke", "anchor", "train-control", "train-e5a", "evaluate"),
        help="依次执行：准备、训练链路检查、E4锚点、对照训练、定向训练、完整验证比较",
    )
    parser.add_argument("--experiment-root", type=Path, help="已创建或待创建的唯一实验目录")
    parser.add_argument("--device", default="0", help="计算设备")
    parser.add_argument("--batch", type=int, default=2, help="推理批量")
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"文档顶层必须是对象：{path}")
    return value


def state_path(root: Path) -> Path:
    return root.resolve() / "experiment_state.json"


def load_state(root: Path) -> dict[str, object]:
    path = state_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"实验状态不存在：{path}")
    return read_json(path)


def save_state(root: Path, state: Mapping[str, object]) -> None:
    write_json_report(state_path(root), dict(state), overwrite=True)


def resolve_experiment_root(requested: Path | None, *, create: bool) -> Path:
    parent = PROJECT_ROOT / "artifacts" / "e5a"
    if requested is None:
        if not create:
            raise ValueError("此阶段必须提供实验目录")
        parent.mkdir(parents=True, exist_ok=True)
        name = allocate_run_name(parent, "e5a_tiny_resampling_001")
        root = parent / name
    else:
        root = requested.resolve()
    if create:
        root.mkdir(parents=True, exist_ok=False)
    elif not root.is_dir():
        raise FileNotFoundError(f"实验目录不存在：{root}")
    return root


def configured_dataset() -> tuple[Path, dict[str, object], Path]:
    if not DEFAULT_DATASET_YAML.is_file():
        raise FileNotFoundError(f"数据配置不存在：{DEFAULT_DATASET_YAML}")
    config = yaml.safe_load(DEFAULT_DATASET_YAML.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("数据配置格式错误")
    root = Path(str(config["path"])).resolve()
    if (root / str(config["train"])).resolve() != (root / "images" / "train").resolve():
        raise ValueError("训练集路径不是冻结的训练目录")
    if (root / str(config["val"])).resolve() != (root / "images" / "val").resolve():
        raise ValueError("验证集路径不是冻结的验证目录")
    return DEFAULT_DATASET_YAML.resolve(), config, root


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def source_fingerprint(dataset_root: Path) -> dict[str, object]:
    cache_files = sorted((dataset_root / "labels").glob("*.cache"))
    return {
        "dataset_yaml_sha256": sha256(DEFAULT_DATASET_YAML),
        "e4_weight_sha256": sha256(E4_WEIGHT),
        "e4_weight_bytes": E4_WEIGHT.stat().st_size,
        "label_caches": {
            str(path.resolve()): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in cache_files
        },
    }


def assert_environment(device: str) -> None:
    import torch
    import ultralytics

    if ultralytics.__version__ != "8.4.120":
        raise RuntimeError(f"Ultralytics版本漂移：{ultralytics.__version__}")
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"计算设备不可用：{device}")


def collect_prediction_records(
    *,
    weight: Path,
    images_dir: Path,
    labels_dir: Path,
    expected_images: int,
    expected_boxes: int,
    device: str,
    batch: int,
) -> list[dict[str, object]]:
    import torch
    from ultralytics import YOLO

    if batch < 1:
        raise ValueError("推理批量必须大于0")
    paths = image_paths(images_dir)
    if len(paths) != expected_images:
        raise RuntimeError(f"图片数量异常：{len(paths)} != {expected_images}")
    expected_ids = {path.name for path in paths}
    source = validated_streaming_image_source(images_dir, expected_images=expected_images)
    model = YOLO(str(weight.resolve()), task="detect")
    if dict(model.names) != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"类别映射异常：{model.names}")
    records: list[dict[str, object]] = []
    results = model.predict(
        source=source,
        imgsz=FIXED_IMGSZ,
        conf=FIXED_CONFIDENCE,
        iou=FIXED_NMS_IOU,
        agnostic_nms=False,
        max_det=FIXED_MAX_DET,
        batch=batch,
        device=device,
        save=False,
        stream=True,
        verbose=False,
    )
    for result in results:
        image_path = Path(result.path).resolve()
        height, width = result.orig_shape
        ground_truth = load_ground_truth_boxes(
            labels_dir / f"{image_path.stem}.txt", image_size=(int(width), int(height))
        )
        predictions = [
            {
                "class_id": int(class_id),
                "box": [float(value) for value in box],
                "confidence": float(confidence),
            }
            for class_id, box, confidence in zip(
                result.boxes.cls.detach().cpu().tolist(),
                result.boxes.xyxy.detach().cpu().tolist(),
                result.boxes.conf.detach().cpu().tolist(),
                strict=True,
            )
        ]
        records.append(
            {
                "image_id": image_path.name,
                "image_path": str(image_path),
                "ground_truth": ground_truth,
                "predictions": predictions,
            }
        )
    del model
    torch.cuda.empty_cache()
    if len(records) != expected_images or {str(row["image_id"]) for row in records} != expected_ids:
        raise RuntimeError("推理结果未完整覆盖指定图片")
    actual_boxes = sum(len(row["ground_truth"]) for row in records)  # type: ignore[arg-type]
    if actual_boxes != expected_boxes:
        raise RuntimeError(f"真实框数量异常：{actual_boxes} != {expected_boxes}")
    return records


def write_manifest(path: Path, entries: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def prepare(root: Path, *, device: str, batch: int) -> dict[str, object]:
    assert_environment(device)
    for required in (E4_WEIGHT, E4_TRAINING_REPORT, E4_VALIDATION_REPORT):
        if not required.is_file():
            raise FileNotFoundError(f"冻结输入不存在：{required}")
    _, _, dataset_root = configured_dataset()
    train_images = image_paths(dataset_root / "images" / "train")
    val_images = image_paths(dataset_root / "images" / "val")
    if len(train_images) != EXPECTED_TRAIN_IMAGES or len(val_images) != EXPECTED_VAL_IMAGES:
        raise RuntimeError("数据清单异常，立即停止")
    before = source_fingerprint(dataset_root)
    started = time.perf_counter()
    records = collect_prediction_records(
        weight=E4_WEIGHT,
        images_dir=dataset_root / "images" / "train",
        labels_dir=dataset_root / "labels" / "train",
        expected_images=EXPECTED_TRAIN_IMAGES,
        expected_boxes=EXPECTED_TRAIN_BOXES,
        device=device,
        batch=batch,
    )
    difficult = identify_tiny_difficult_samples(
        records, max_equivalent_size=10.0, iou_threshold=FIXED_MATCHING_IOU
    )
    del records
    if not difficult:
        raise RuntimeError("训练集没有识别出微小目标困难样本")
    manifests = build_resampled_manifests(
        train_images, difficult, extra_count=EXTRA_ENTRIES, seed=42
    )
    original_audits = {
        "control": audit_training_manifest(
            manifests["control_entries"],
            original_train_images=train_images,
            processed_root=dataset_root,
            expected_entries=FORMAL_ENTRIES,
        ),
        "e5a": audit_training_manifest(
            manifests["e5a_entries"],
            original_train_images=train_images,
            processed_root=dataset_root,
            expected_entries=FORMAL_ENTRIES,
        ),
    }
    manifests_dir = root / "manifests"
    write_manifest(manifests_dir / "control_6500_original_paths.txt", manifests["control_entries"])
    write_manifest(manifests_dir / "e5a_6500_original_paths.txt", manifests["e5a_entries"])
    write_manifest(manifests_dir / "control_extra_1043.txt", manifests["control_extra"])
    write_manifest(manifests_dir / "e5a_extra_1043.txt", manifests["e5a_extra"])
    difficult_json = manifests_dir / "train_tiny_difficult_samples.json"
    write_json_report(
        difficult_json,
        {
            "source_split": "train",
            "images": len(difficult),
            "tiny_missed": sum(int(row["tiny_missed"]) for row in difficult),
            "tiny_helmet_missed": sum(int(row["tiny_helmet_missed"]) for row in difficult),
            "tiny_no_helmet_missed": sum(int(row["tiny_no_helmet_missed"]) for row in difficult),
            "sampling_weight_formula": "2*tiny_helmet_missed + tiny_no_helmet_missed",
            "rows": difficult,
        },
    )
    difficult_csv = manifests_dir / "train_tiny_difficult_samples.csv"
    with difficult_csv.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(difficult[0]))
        writer.writeheader()
        writer.writerows(difficult)
    staged = {
        "control": stage_dataset_variant(
            source_root=dataset_root,
            stage_root=root / "datasets" / "control",
            manifest_entries=manifests["control_entries"],
        ),
        "e5a": stage_dataset_variant(
            source_root=dataset_root,
            stage_root=root / "datasets" / "e5a",
            manifest_entries=manifests["e5a_entries"],
        ),
    }
    after = source_fingerprint(dataset_root)
    if after != before:
        raise RuntimeError("原数据或E4权重在准备阶段发生变化")
    report = {
        "status": "passed",
        "phase": "prepare",
        "created_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "experiment_root": str(root),
        "protocol": {
            "source_weight": str(E4_WEIGHT.resolve()),
            "source_weight_sha256": before["e4_weight_sha256"],
            "imgsz": FIXED_IMGSZ,
            "confidence": FIXED_CONFIDENCE,
            "nms_iou": FIXED_NMS_IOU,
            "matching_iou": FIXED_MATCHING_IOU,
            "class_aware_nms": True,
            "class_aware_matching": True,
            "tiny_max_equivalent_size": 10.0,
            "all_ground_truth_loaded_for_matching": True,
            "sampling_seed": 42,
            "sampling_with_replacement": True,
            "helmet_miss_weight": 2,
            "no_helmet_miss_weight": 1,
            "validation_used_for_sampling": False,
            "held_out_data_used": False,
        },
        "difficult_summary": read_json(difficult_json),
        "original_manifest_audits": original_audits,
        "staged_datasets": staged,
        "source_fingerprint_before": before,
        "source_fingerprint_after": after,
    }
    write_json_report(root / "preparation_audit.json", report)
    difficult_summary = report["difficult_summary"]  # type: ignore[assignment]
    markdown = "\n".join(
        [
            "# E5-A准备与抽样审计",
            "",
            f"- 训练集图片：{EXPECTED_TRAIN_IMAGES}张；两份训练清单：各{FORMAL_ENTRIES}条。",
            f"- 微小困难图片：{difficult_summary['images']}张；微小目标漏检：{difficult_summary['tiny_missed']}个。",
            f"- 其中安全帽漏检：{difficult_summary['tiny_helmet_missed']}个；未戴安全帽漏检：{difficult_summary['tiny_no_helmet_missed']}个。",
            "- 抽样权重：微小安全帽漏检数乘2，加微小未戴安全帽漏检数乘1；固定随机种子42，有放回抽样。",
            "- 匹配时加载每张训练图片中的全部真实框，微小目标按原图框等效边长不超过10像素定义。",
            "- 验证图片未用于困难样本识别或抽样，独立留出数据未使用。",
            "- 原数据、E4权重及原缓存指纹前后一致。",
            "",
        ]
    )
    (root / "preparation_audit.md").write_text(markdown, encoding="utf-8")
    state = {
        "status": "active",
        "experiment_root": str(root),
        "created_at_utc": utc_now(),
        "source_weight": str(E4_WEIGHT.resolve()),
        "source_weight_sha256": sha256(E4_WEIGHT),
        "dataset_root": str(dataset_root),
        "source_fingerprint": before,
        "prepare_passed": True,
        "smoke_passed": False,
        "anchor_passed": False,
        "control_training_passed": False,
        "e5a_training_passed": False,
        "staged_datasets": staged,
        "held_out_data_used": False,
    }
    save_state(root, state)
    return report


def _label_classes(label_path: Path) -> set[int]:
    return {
        int(line.split()[0])
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_smoke_dataset(root: Path, source_root: Path) -> Path:
    smoke_root = root / "smoke" / "dataset"
    if smoke_root.exists():
        raise FileExistsError(f"训练链路检查数据已存在：{smoke_root}")
    source_images = image_paths(source_root / "images" / "train")
    helmet = [p for p in source_images if 0 in _label_classes(source_root / "labels" / "train" / f"{p.stem}.txt")]
    no_helmet = [p for p in source_images if 1 in _label_classes(source_root / "labels" / "train" / f"{p.stem}.txt")]
    selected_helmet = helmet[:6]
    selected_no_helmet = [path for path in no_helmet if path not in set(selected_helmet)][:6]
    if len(selected_helmet) != 6 or len(selected_no_helmet) != 6:
        raise RuntimeError("无法从训练集构造包含两个类别的链路检查数据")
    train = [*selected_helmet[:4], *selected_no_helmet[:4]]
    val = [*selected_helmet[4:], *selected_no_helmet[4:]]
    for split, paths in (("train", train), ("val", val)):
        images_destination = smoke_root / "images" / split
        labels_destination = smoke_root / "labels" / split
        images_destination.mkdir(parents=True, exist_ok=False)
        labels_destination.mkdir(parents=True, exist_ok=False)
        for image in paths:
            os.link(image, images_destination / image.name)
            shutil.copy2(
                source_root / "labels" / "train" / f"{image.stem}.txt",
                labels_destination / f"{image.stem}.txt",
            )
    manifest_entries = [str((smoke_root / "images" / "train" / path.name).resolve()) for path in train]
    manifest_entries.extend(manifest_entries[:2])
    manifest = smoke_root / "train_with_duplicates.txt"
    write_manifest(manifest, manifest_entries)
    dataset_yaml = smoke_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(smoke_root.resolve()),
                "train": str(manifest.resolve()),
                "val": "images/val",
                "names": {0: "helmet", 1: "no_helmet"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def smoke(root: Path, *, device: str) -> dict[str, object]:
    assert_environment(device)
    state = load_state(root)
    if not state.get("prepare_passed"):
        raise RuntimeError("准备与抽样审计尚未通过")
    source_root = Path(str(state["dataset_root"]))
    dataset_yaml = build_smoke_dataset(root, source_root)
    e4_training = read_json(E4_TRAINING_REPORT)
    e4_args = e4_training["training_parameters"]
    if not isinstance(e4_args, Mapping):
        raise ValueError("E4训练参数缺失")
    from ultralytics import YOLO

    kwargs = build_e5a_training_kwargs(
        e4_args=e4_args,
        data_yaml=dataset_yaml,
        project_dir=root / "smoke" / "training",
        run_name="chain_check",
        device=device,
    )
    kwargs.update({"epochs": 1, "patience": 1, "imgsz": 320, "plots": False})
    started = time.perf_counter()
    model = YOLO(str(E4_WEIGHT.resolve()), task="detect")
    model.train(**kwargs)
    save_dir = Path(model.trainer.save_dir).resolve()
    loaded_entries = len(model.trainer.train_loader.dataset.im_files)
    del model
    if loaded_entries != 10:
        raise RuntimeError(f"重复清单链路检查失败：实际加载{loaded_entries}条，预期10条")
    results_csv = save_dir / "results.csv"
    args_yaml = yaml.safe_load((save_dir / "args.yaml").read_text(encoding="utf-8"))
    if not results_csv.is_file() or len(results_csv.read_text(encoding="utf-8").splitlines()) != 2:
        raise RuntimeError("小规模训练没有完成一个训练周期")
    if args_yaml.get("resume") is not False:
        raise RuntimeError("小规模训练错误地启用了恢复训练")
    report = {
        "status": "passed",
        "phase": "smoke",
        "source_split": "train",
        "original_validation_images_used": False,
        "loaded_training_entries": loaded_entries,
        "completed_epochs": 1,
        "resume": False,
        "start_weight_sha256": sha256(E4_WEIGHT),
        "save_dir": str(save_dir),
        "duration_seconds": time.perf_counter() - started,
    }
    write_json_report(root / "smoke" / "smoke_report.json", report)
    (root / "smoke" / "smoke_report.md").write_text(
        "# 小规模训练链路检查\n\n"
        "- 使用训练集派生的8张训练图片和4张链路验证图片，未使用原验证集。\n"
        "- 重复训练清单共10条，框架实际加载10条。\n"
        "- 从E4最佳权重独立开始，恢复训练标志为否，完成1个训练周期。\n",
        encoding="utf-8",
    )
    state["smoke_passed"] = True
    state["smoke_report"] = report
    save_state(root, state)
    return report


def anchor(root: Path, *, device: str, batch: int) -> dict[str, object]:
    assert_environment(device)
    state = load_state(root)
    if not state.get("prepare_passed") or not state.get("smoke_passed"):
        raise RuntimeError("准备审计和小规模训练检查必须先通过")
    dataset_root = Path(str(state["dataset_root"]))
    records = collect_prediction_records(
        weight=E4_WEIGHT,
        images_dir=dataset_root / "images" / "val",
        labels_dir=dataset_root / "labels" / "val",
        expected_images=EXPECTED_VAL_IMAGES,
        expected_boxes=EXPECTED_VAL_BOXES,
        device=device,
        batch=batch,
    )
    metrics = summarize_e5a_evaluation(records, matching_iou=FIXED_MATCHING_IOU)
    anchor_counts = validate_e4_candidate_c_anchor(metrics)
    report = {
        "status": "passed",
        "phase": "anchor",
        "created_at_utc": utc_now(),
        "protocol": {
            "images": EXPECTED_VAL_IMAGES,
            "ground_truth": EXPECTED_VAL_BOXES,
            "imgsz": FIXED_IMGSZ,
            "confidence": FIXED_CONFIDENCE,
            "nms_iou": FIXED_NMS_IOU,
            "matching_iou": FIXED_MATCHING_IOU,
            "class_aware": True,
            "held_out_data_used": False,
        },
        "anchor_counts": anchor_counts,
        "metrics": metrics,
    }
    output = root / "anchor"
    output.mkdir(parents=True, exist_ok=False)
    write_json_report(output / "e4_candidate_c_anchor.json", report)
    (output / "e4_candidate_c_anchor.md").write_text(
        render_chinese_evaluation_markdown({"E4候选C": metrics})
        + "\n完整607张验证集锚点复现通过；独立留出数据未使用。\n",
        encoding="utf-8",
    )
    state["anchor_passed"] = True
    state["anchor_report"] = report
    save_state(root, state)
    return report


def _assert_formal_args(args_yaml: Mapping[str, object]) -> None:
    expected = {
        "epochs": 30,
        "patience": 10,
        "imgsz": 960,
        "batch": 2,
        "workers": 0,
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
        "seed": 42,
        "deterministic": True,
        "amp": True,
        "resume": False,
    }
    actual = {key: args_yaml.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"正式训练参数漂移：实际{actual}，预期{expected}")


def formal_train(root: Path, *, variant: str, device: str) -> dict[str, object]:
    assert_environment(device)
    if variant not in {"control", "e5a"}:
        raise ValueError(f"未知训练组：{variant}")
    state = load_state(root)
    if not all(state.get(key) for key in ("prepare_passed", "smoke_passed", "anchor_passed")):
        raise RuntimeError("正式训练门禁未全部通过")
    state_key = "control_training_passed" if variant == "control" else "e5a_training_passed"
    if state.get(state_key):
        raise RuntimeError("该训练组已经完成，禁止覆盖")
    if sha256(E4_WEIGHT) != state["source_weight_sha256"]:
        raise RuntimeError("E4起始权重指纹发生变化")
    staged = state["staged_datasets"]
    if not isinstance(staged, Mapping) or not isinstance(staged[variant], Mapping):
        raise ValueError("派生数据状态缺失")
    dataset_yaml = Path(str(staged[variant]["dataset_yaml"]))
    e4_training = read_json(E4_TRAINING_REPORT)
    e4_args = e4_training["training_parameters"]
    if not isinstance(e4_args, Mapping):
        raise ValueError("E4训练参数缺失")
    run_name = "extended_training_control" if variant == "control" else "e5a_targeted_resampling"
    training_project = root / "training"
    kwargs = build_e5a_training_kwargs(
        e4_args=e4_args,
        data_yaml=dataset_yaml,
        project_dir=training_project,
        run_name=run_name,
        device=device,
    )
    from ultralytics import YOLO

    started_at = utc_now()
    started = time.perf_counter()
    model = YOLO(str(E4_WEIGHT.resolve()), task="detect")
    model.train(**kwargs)
    save_dir = Path(model.trainer.save_dir).resolve()
    del model
    args_yaml = yaml.safe_load((save_dir / "args.yaml").read_text(encoding="utf-8"))
    _assert_formal_args(args_yaml)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    results_csv = save_dir / "results.csv"
    for path in (best_pt, last_pt, results_csv):
        if not path.is_file():
            raise FileNotFoundError(f"正式训练产物缺失：{path}")
    analysis = analyze_training_results(results_csv, requested_epochs=30)
    report = {
        "status": "passed",
        "phase": f"train-{variant}",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "source_weight": str(E4_WEIGHT.resolve()),
        "source_weight_sha256": sha256(E4_WEIGHT),
        "resume": False,
        "dataset_yaml": str(dataset_yaml.resolve()),
        "formal_parameters": kwargs,
        "actual_args": args_yaml,
        "training_analysis": analysis,
        "outputs": {
            "save_dir": str(save_dir),
            "best_pt": str(best_pt.resolve()),
            "best_pt_sha256": sha256(best_pt),
            "last_pt": str(last_pt.resolve()),
            "last_pt_sha256": sha256(last_pt),
            "results_csv": str(results_csv.resolve()),
        },
        "original_validation_used_for_sampling": False,
        "held_out_data_used": False,
    }
    write_json_report(save_dir / "e5a_training_report.json", report)
    chinese_name = "延长训练对照" if variant == "control" else "E5-A定向重采样"
    (save_dir / "e5a_training_report.md").write_text(
        "# " + chinese_name + "训练报告\n\n"
        "- 从E4最佳权重独立开始，恢复训练标志为否。\n"
        f"- 实际完成训练周期：{analysis['epochs_completed']}；最佳周期：{analysis['best_epoch']}。\n"
        f"- 训练耗时：{report['duration_seconds'] / 3600:.3f}小时。\n"
        "- 原验证图片未参与困难样本识别或抽样；独立留出数据未使用。\n",
        encoding="utf-8",
    )
    state[state_key] = True
    state[f"{variant}_training_report"] = report
    save_state(root, state)
    return report


def _metric_deltas(control: Mapping[str, object], e5a: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scopes = (("overall", "总体"), ("helmet", "安全帽"), ("no_helmet", "未戴安全帽"), ("tiny", "微小目标"))
    control_per_class = control["per_class"]  # type: ignore[assignment]
    e5a_per_class = e5a["per_class"]  # type: ignore[assignment]
    for key, name in scopes:
        first = control[key] if key in {"overall", "tiny"} else control_per_class[key]  # type: ignore[index]
        second = e5a[key] if key in {"overall", "tiny"} else e5a_per_class[key]  # type: ignore[index]
        rows.append(
            {
                "scope": name,
                "correct_detections": int(second["tp"]) - int(first["tp"]),
                "misses": int(second["fn"]) - int(first["fn"]),
                "false_detections": int(second["fp"]) - int(first["fp"]),
                "precision": float(second["precision"]) - float(first["precision"]),
                "recall": float(second["recall"]) - float(first["recall"]),
                "f1": float(second["f1"]) - float(first["f1"]),
            }
        )
    return rows


def evaluate(root: Path, *, device: str, batch: int) -> dict[str, object]:
    assert_environment(device)
    state = load_state(root)
    if not all(state.get(key) for key in ("control_training_passed", "e5a_training_passed", "anchor_passed")):
        raise RuntimeError("两组正式训练和E4锚点必须全部完成")
    dataset_root = Path(str(state["dataset_root"]))
    if source_fingerprint(dataset_root) != state["source_fingerprint"]:
        raise RuntimeError("原数据、缓存或E4权重指纹发生变化")
    anchor_report = state["anchor_report"]
    if not isinstance(anchor_report, Mapping):
        raise ValueError("E4锚点状态缺失")
    e4_metrics = anchor_report["metrics"]
    reports = {
        "control": state["control_training_report"],
        "e5a": state["e5a_training_report"],
    }
    metrics: dict[str, Mapping[str, object]] = {"E4": e4_metrics}  # type: ignore[dict-item]
    for variant, display in (("control", "延长训练对照"), ("e5a", "E5-A")):
        training_report = reports[variant]
        if not isinstance(training_report, Mapping):
            raise ValueError(f"训练报告缺失：{variant}")
        outputs = training_report["outputs"]
        if not isinstance(outputs, Mapping):
            raise ValueError(f"训练权重路径缺失：{variant}")
        best_pt = Path(str(outputs["best_pt"]))
        if sha256(best_pt) != outputs["best_pt_sha256"]:
            raise RuntimeError(f"最佳权重指纹异常：{variant}")
        records = collect_prediction_records(
            weight=best_pt,
            images_dir=dataset_root / "images" / "val",
            labels_dir=dataset_root / "labels" / "val",
            expected_images=EXPECTED_VAL_IMAGES,
            expected_boxes=EXPECTED_VAL_BOXES,
            device=device,
            batch=batch,
        )
        metrics[display] = summarize_e5a_evaluation(records, matching_iou=FIXED_MATCHING_IOU)
    deltas = _metric_deltas(metrics["延长训练对照"], metrics["E5-A"])
    judgment = judge_resampling_effectiveness(metrics["延长训练对照"], metrics["E5-A"])
    report = {
        "status": "passed",
        "phase": "evaluate",
        "created_at_utc": utc_now(),
        "protocol": {
            "images": EXPECTED_VAL_IMAGES,
            "ground_truth": EXPECTED_VAL_BOXES,
            "imgsz": FIXED_IMGSZ,
            "confidence": FIXED_CONFIDENCE,
            "nms_iou": FIXED_NMS_IOU,
            "matching_iou": FIXED_MATCHING_IOU,
            "class_aware_nms": True,
            "class_aware_matching": True,
            "tiny_max_equivalent_size": 10.0,
            "tiny_false_detection_definition": "unmatched predictions with equivalent size <= 10 pixels",
            "held_out_data_used": False,
        },
        "metrics": metrics,
        "e5a_deltas_vs_control": deltas,
        "effectiveness_judgment": judgment,
    }
    output = root / allocate_run_name(root, "evaluation")
    output.mkdir(parents=True, exist_ok=False)
    write_json_report(
        output / "e5a_full_validation_comparison.json",
        chinese_detection_count_keys(report),
    )
    markdown = render_chinese_evaluation_markdown(metrics)
    markdown += "\n## E5-A相对延长训练对照的变化\n\n"
    markdown += "| 范围 | 正确检出变化 | 漏检变化 | 误检变化 | 查准率变化 | 召回率变化 | 综合指标变化 |\n"
    markdown += "|---|---:|---:|---:|---:|---:|---:|\n"
    for row in deltas:
        markdown += (
            f"| {row['scope']} | {int(row['correct_detections']):+d} | {int(row['misses']):+d} | "
            f"{int(row['false_detections']):+d} | {float(row['precision']):+.6f} | "
            f"{float(row['recall']):+.6f} | {float(row['f1']):+.6f} |\n"
        )
    conclusion = "重采样真正有效" if judgment["effective"] else "现有证据不足以证明重采样真正有效"
    markdown += "\n## 判断\n\n"
    markdown += f"- 结论：{conclusion}。\n"
    markdown += "- 预先声明的判据：微小目标召回率和综合指标均提高，且总体综合指标下降不超过0.002。\n"
    markdown += "- 微小目标误检定义为完整验证集上未匹配且预测框等效边长不超过10像素的预测。\n"
    markdown += "- 独立留出数据未使用。\n"
    (output / "e5a_full_validation_comparison.md").write_text(markdown, encoding="utf-8")
    state["status"] = "passed"
    state["evaluation_passed"] = True
    state["evaluation_report"] = report
    save_state(root, state)
    return report


def run(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    root = resolve_experiment_root(args.experiment_root, create=args.phase == "prepare")
    if args.phase == "prepare":
        return root, prepare(root, device=args.device, batch=args.batch)
    if args.phase == "smoke":
        return root, smoke(root, device=args.device)
    if args.phase == "anchor":
        return root, anchor(root, device=args.device, batch=args.batch)
    if args.phase == "train-control":
        return root, formal_train(root, variant="control", device=args.device)
    if args.phase == "train-e5a":
        return root, formal_train(root, variant="e5a", device=args.device)
    return root, evaluate(root, device=args.device, batch=args.batch)


def main() -> int:
    try:
        root, report = run(build_parser().parse_args())
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"status": report["status"], "phase": report["phase"], "experiment_root": str(root)},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
