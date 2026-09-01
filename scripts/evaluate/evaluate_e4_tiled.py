#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

from PIL import Image
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.inference.tiling import (  # noqa: E402
    merge_hybrid_predictions,
    predict_tiled_image,
)
from helmet_safety.training.baseline import (  # noqa: E402
    allocate_run_name,
    load_ground_truth_boxes,
    write_json_report,
)
from helmet_safety.training.analysis_core import (  # noqa: E402
    audit_obvious_duplicate_boxes,
    evaluate_tiny_conf_records,
    summarize_fixed_threshold_detections,
)


FIXED_WEIGHT = (
    PROJECT_ROOT
    / "artifacts"
    / "training"
    / "m45_yolo11s_e75_960_001"
    / "weights"
    / "best.pt"
)
FIXED_TINY_MANIFEST = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation"
    / "m45_yolo11s_e75_960_tiny_val_001"
    / "tiny_val_images.json"
)
FIXED_ORIGINAL_FN = FIXED_TINY_MANIFEST.with_name("tiny_val_false_negatives.json")
FIXED_P0_REPORT = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation"
    / "m45_yolo11s_e75_960_full_val_p0_001"
    / "e4_full_val_p0_report.json"
)
FIXED_IMGSZ = 960
FIXED_CONFIDENCE = 0.20
FIXED_NMS_IOU = 0.50
FIXED_MATCHING_IOU = 0.50
FIXED_OVERLAP = 0.20
FIXED_MAX_DET = 300
TILE_CONFIGS = (("S1_hybrid_768", 768), ("S2_hybrid_640", 640))
EXPECTED_TINY_IMAGES = 35
EXPECTED_TINY_GT = 128
EXPECTED_ORIGINAL_FN = 51
EXPECTED_VAL_IMAGES = 607
EXPECTED_VAL_GT = 9925
EXPECTED_C_FULL = {"tp": 9420, "fn": 505, "fp": 951}
EXPECTED_C_TINY = {
    "tp": 79,
    "fn": 49,
    "helmet_tp": 5,
    "helmet_fn": 9,
    "no_helmet_tp": 74,
    "no_helmet_fn": 40,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate E4 best.pt on val only with Candidate C (imgsz=960, conf=0.20, "
            "class-aware NMS IoU=0.50, matching IoU=0.5) and fixed hybrid "
            "tiles=768/640 at overlap=0.20. Test is never used."
        )
    )
    parser.add_argument("--scope", choices=("tiny", "full"), default="tiny")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--output-name")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_dicts(result: object) -> list[dict[str, object]]:
    boxes = result.boxes  # type: ignore[attr-defined]
    return [
        {
            "class_id": int(class_id),
            "confidence": float(confidence),
            "box": [float(value) for value in box],
        }
        for class_id, confidence, box in zip(
            boxes.cls.detach().cpu().tolist(),
            boxes.conf.detach().cpu().tolist(),
            boxes.xyxy.detach().cpu().tolist(),
            strict=True,
        )
    ]


def candidate_c_tiny_anchor(row: Mapping[str, object]) -> dict[str, int]:
    """Extract the frozen Candidate C tiny counts from one comparison row."""

    return {
        "tp": int(row["tiny_tp"]),
        "fn": int(row["tiny_fn"]),
        "helmet_tp": int(row["tiny_helmet_tp"]),
        "helmet_fn": int(row["tiny_helmet_fn"]),
        "no_helmet_tp": int(row["tiny_no_helmet_tp"]),
        "no_helmet_fn": int(row["tiny_no_helmet_fn"]),
    }


def tile_configs_for_scope(scope: str) -> tuple[tuple[str, int], ...]:
    """Screen both tile sizes on tiny val, then promote only 640 to full val."""

    if scope == "tiny":
        return TILE_CONFIGS
    if scope == "full":
        return (TILE_CONFIGS[1],)
    raise ValueError(f"unsupported scope: {scope}")


def _markdown(rows: Sequence[Mapping[str, object]], *, scope: str) -> str:
    lines = [
        f"# E4 切片/滑窗推理（{scope} val）",
        "",
        "- 权重：E4 best.pt；未训练、未使用 test。",
        "- 固定：imgsz=960、conf=0.20、class-aware NMS IoU=0.50、matching IoU=0.5、max_det=300。",
        "- S1/S2：Candidate C 整图预测与 20% 重叠切片预测回写原图后，再执行一次全局 class-aware NMS。",
        "",
        "| 配置 | tile | TP | FN | FP | Precision | Recall | F1 | tiny TP | tiny FN | tiny R | tiny helmet R | tiny no_helmet R | 救回原51 FN | 重复框对 | 前向图块 | ms/图 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        tile = "—" if row["tile_size"] is None else str(row["tile_size"])
        lines.append(
            f"| {row['config']} | {tile} | {row['overall_tp']} | {row['overall_fn']} | "
            f"{row['overall_fp']} | {float(row['overall_precision']):.6f} | "
            f"{float(row['overall_recall']):.6f} | {float(row['overall_f1']):.6f} | "
            f"{row['tiny_tp']} | {row['tiny_fn']} | {float(row['tiny_recall']):.6f} | "
            f"{float(row['tiny_helmet_recall']):.6f} | "
            f"{float(row['tiny_no_helmet_recall']):.6f} | "
            f"{row['recovered_original_fn_count']} | {row['duplicate_pairs']} | "
            f"{row['forward_images']} | {float(row['wall_ms_per_image']):.3f} |"
        )
    lines.extend(["", "## 原51个 FN 的救回", ""])
    for row in rows:
        keys = str(row["recovered_original_fn_keys"]) or "无"
        lines.append(f"- {row['config']}：{keys}")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    if args.batch < 1:
        raise ValueError("batch must be at least 1")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} unavailable")
    fixed_inputs = (FIXED_WEIGHT, FIXED_TINY_MANIFEST, FIXED_ORIGINAL_FN, FIXED_P0_REPORT)
    for path in fixed_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"fixed input is unavailable: {path.resolve()}")
    hashes_before = {path.name: _sha256(path) for path in fixed_inputs}

    training_args = yaml.safe_load((FIXED_WEIGHT.parents[2] / "args.yaml").read_text(encoding="utf-8"))
    dataset_yaml = Path(str(training_args["data"])).resolve()
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    processed_root = Path(str(dataset["path"])).resolve()
    val_images_dir = (processed_root / str(dataset["val"])).resolve()
    val_labels_dir = (processed_root / "labels" / "val").resolve()
    all_image_paths = sorted(
        path.resolve()
        for path in val_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(all_image_paths) != EXPECTED_VAL_IMAGES:
        raise RuntimeError(f"val image count mismatch: {len(all_image_paths)} != {EXPECTED_VAL_IMAGES}")

    manifest = json.loads(FIXED_TINY_MANIFEST.read_text(encoding="utf-8"))
    if int(manifest["image_count"]) != EXPECTED_TINY_IMAGES or int(manifest["tiny_gt_count"]) != EXPECTED_TINY_GT:
        raise RuntimeError("frozen tiny manifest no longer matches 35 images / 128 tiny GT")
    tiny_ids = [str(item["image_id"]) for item in manifest["images"]]
    image_by_id = {path.name: path for path in all_image_paths}
    image_paths = (
        [image_by_id[image_id] for image_id in tiny_ids]
        if args.scope == "tiny"
        else all_image_paths
    )
    original_fn_doc = json.loads(FIXED_ORIGINAL_FN.read_text(encoding="utf-8"))
    original_fns = original_fn_doc["false_negatives"]
    if len(original_fns) != EXPECTED_ORIGINAL_FN:
        raise RuntimeError(f"original FN count mismatch: {len(original_fns)} != {EXPECTED_ORIGINAL_FN}")

    ground_truth_by_image: dict[str, list[dict[str, object]]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_size = image.size
        ground_truth_by_image[image_path.name] = load_ground_truth_boxes(
            val_labels_dir / f"{image_path.stem}.txt", image_size=image_size
        )
    if args.scope == "full":
        gt_count = sum(len(items) for items in ground_truth_by_image.values())
        if gt_count != EXPECTED_VAL_GT:
            raise RuntimeError(f"full val GT count mismatch: {gt_count} != {EXPECTED_VAL_GT}")

    model = YOLO(str(FIXED_WEIGHT.resolve()), task="detect")
    if dict(model.names) != {0: "helmet", 1: "no_helmet"}:
        raise ValueError(f"unexpected class mapping: {model.names}")
    predict_kwargs = {
        "imgsz": FIXED_IMGSZ,
        "conf": FIXED_CONFIDENCE,
        "iou": FIXED_NMS_IOU,
        "agnostic_nms": False,
        "max_det": FIXED_MAX_DET,
        "rect": False,
        "device": args.device,
        "save": False,
        "verbose": False,
    }
    model.predict(source=str(image_paths[0]), batch=1, **predict_kwargs)

    total_started = time.perf_counter()
    full_started = time.perf_counter()
    full_source: object = (
        str(val_images_dir) if args.scope == "full" else [str(path) for path in image_paths]
    )
    full_results = model.predict(
        source=full_source,
        batch=args.batch,
        stream=True,
        **predict_kwargs,
    )
    full_predictions: dict[str, list[dict[str, object]]] = {}
    for result in full_results:
        full_predictions[Path(result.path).name] = _prediction_dicts(result)
    full_seconds = time.perf_counter() - full_started
    if set(full_predictions) != set(ground_truth_by_image):
        raise RuntimeError("full-image inference did not return the selected val images exactly once")

    records_by_config: dict[str, list[dict[str, object]]] = {
        "S0_candidate_C": [
            {
                "image_id": image_path.name,
                "ground_truth": ground_truth_by_image[image_path.name],
                "predictions": full_predictions[image_path.name],
            }
            for image_path in image_paths
        ]
    }
    tile_counts = {"S0_candidate_C": 0}
    tile_seconds = {"S0_candidate_C": 0.0}
    active_tile_configs = tile_configs_for_scope(args.scope)

    def predict_batch(crops: list[Image.Image]) -> list[list[dict[str, object]]]:
        results = model.predict(source=crops, batch=args.batch, **predict_kwargs)
        return [_prediction_dicts(result) for result in results]

    for config_name, tile_size in active_tile_configs:
        config_started = time.perf_counter()
        tile_count = 0
        records: list[dict[str, object]] = []
        for index, image_path in enumerate(image_paths, start=1):
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            try:
                tiled = predict_tiled_image(
                    image,
                    tile_size=tile_size,
                    overlap_ratio=FIXED_OVERLAP,
                    batch_size=args.batch,
                    predict_batch=predict_batch,
                )
            finally:
                image.close()
            tile_count += int(tiled["tile_count"])
            merged = merge_hybrid_predictions(
                full_predictions[image_path.name],
                tiled["predictions"],  # type: ignore[arg-type]
                nms_iou=FIXED_NMS_IOU,
                max_detections=FIXED_MAX_DET,
            )
            records.append(
                {
                    "image_id": image_path.name,
                    "ground_truth": ground_truth_by_image[image_path.name],
                    "predictions": merged,
                }
            )
            if index % 25 == 0 or index == len(image_paths):
                print(
                    json.dumps(
                        {
                            "config": config_name,
                            "processed": index,
                            "images": len(image_paths),
                            "tiles": tile_count,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        records_by_config[config_name] = records
        tile_counts[config_name] = tile_count
        tile_seconds[config_name] = time.perf_counter() - config_started

    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    config_specs = (("S0_candidate_C", None), *active_tile_configs)
    for config_name, tile_size in config_specs:
        records = records_by_config[config_name]
        fixed = summarize_fixed_threshold_detections(records, iou_threshold=FIXED_MATCHING_IOU)
        tiny_evaluation = evaluate_tiny_conf_records(
            records,
            original_fns,
            iou_threshold=FIXED_MATCHING_IOU,
        )
        duplicates = audit_obvious_duplicate_boxes(records, duplicate_iou_threshold=0.70)
        overall = fixed["overall"]
        helmet = fixed["per_class"]["helmet"]
        no_helmet = fixed["per_class"]["no_helmet"]
        tiny = tiny_evaluation["summary"]
        if int(overall["fp"]) != int(tiny_evaluation["false_positives"]):
            raise RuntimeError(f"FP accounting mismatch for {config_name}")
        elapsed = full_seconds + tile_seconds[config_name]
        recovered = tiny_evaluation["recovered_original_false_negatives"]
        rows.append(
            {
                "config": config_name,
                "tile_size": tile_size,
                "overlap_ratio": None if tile_size is None else FIXED_OVERLAP,
                "images": len(image_paths),
                "tile_count": tile_counts[config_name],
                "forward_images": len(image_paths) + tile_counts[config_name],
                "overall_tp": overall["tp"],
                "overall_fn": overall["fn"],
                "overall_fp": overall["fp"],
                "overall_precision": overall["precision"],
                "overall_recall": overall["recall"],
                "overall_f1": overall["f1"],
                "helmet_tp": helmet["tp"],
                "helmet_fn": helmet["fn"],
                "helmet_fp": helmet["fp"],
                "helmet_recall": helmet["recall"],
                "no_helmet_tp": no_helmet["tp"],
                "no_helmet_fn": no_helmet["fn"],
                "no_helmet_fp": no_helmet["fp"],
                "no_helmet_recall": no_helmet["recall"],
                "tiny_tp": tiny["tp"],
                "tiny_fn": tiny["fn"],
                "tiny_recall": tiny["recall"],
                "tiny_helmet_tp": tiny["helmet_tp"],
                "tiny_helmet_fn": tiny["helmet_fn"],
                "tiny_helmet_recall": tiny["helmet_recall"],
                "tiny_no_helmet_tp": tiny["no_helmet_tp"],
                "tiny_no_helmet_fn": tiny["no_helmet_fn"],
                "tiny_no_helmet_recall": tiny["no_helmet_recall"],
                "recovered_original_fn_count": tiny_evaluation["recovered_original_fn_count"],
                "recovered_helmet_count": tiny_evaluation["recovered_helmet_count"],
                "recovered_no_helmet_count": tiny_evaluation["recovered_no_helmet_count"],
                "recovered_original_fn_keys": "; ".join(
                    f"{item['image_id']}:{item['gt_index']}:{item['class_name']}"
                    for item in recovered
                ),
                "duplicate_pairs": duplicates["duplicate_pairs"],
                "images_with_duplicates": duplicates["images_with_duplicates"],
                "wall_seconds": elapsed,
                "wall_ms_per_image": elapsed * 1000.0 / len(image_paths),
            }
        )
        details[config_name] = {
            "recovered_original_false_negatives": recovered,
            "duplicate_audit": duplicates,
        }

    baseline = rows[0]
    actual_tiny = candidate_c_tiny_anchor(baseline)
    if actual_tiny != EXPECTED_C_TINY:
        raise RuntimeError(f"Candidate C tiny anchor mismatch: {actual_tiny} != {EXPECTED_C_TINY}")
    if args.scope == "full":
        actual_full = {
            "tp": int(baseline["overall_tp"]),
            "fn": int(baseline["overall_fn"]),
            "fp": int(baseline["overall_fp"]),
        }
        if actual_full != EXPECTED_C_FULL:
            raise RuntimeError(f"Candidate C full-val anchor mismatch: {actual_full} != {EXPECTED_C_FULL}")

    hashes_after = {path.name: _sha256(path) for path in fixed_inputs}
    if hashes_after != hashes_before:
        raise RuntimeError("a fixed E4 input changed during tiled evaluation")

    evaluation_root = PROJECT_ROOT / "artifacts" / "evaluation"
    requested_name = args.output_name or f"m45_yolo11s_e75_960_tiled_{args.scope}_val_001"
    run_name = allocate_run_name(evaluation_root, requested_name)
    output_dir = evaluation_root / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "e4_tiled_comparison.csv"
    markdown_path = output_dir / "e4_tiled_comparison.md"
    details_path = output_dir / "e4_tiled_details.json"
    report_path = output_dir / "e4_tiled_report.json"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(_markdown(rows, scope=args.scope), encoding="utf-8")
    write_json_report(details_path, details)
    report = {
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "weight": str(FIXED_WEIGHT.resolve()),
            "weight_sha256": hashes_before[FIXED_WEIGHT.name],
            "dataset_yaml": str(dataset_yaml),
            "scope": args.scope,
            "split": "val",
            "test_evaluated": False,
            "imgsz": FIXED_IMGSZ,
            "confidence": FIXED_CONFIDENCE,
            "local_nms_iou": FIXED_NMS_IOU,
            "global_nms_iou": FIXED_NMS_IOU,
            "class_aware_nms": True,
            "matching_iou": FIXED_MATCHING_IOU,
            "tile_configs": [
                {"config": name, "tile_size": size, "overlap_ratio": FIXED_OVERLAP}
                for name, size in active_tile_configs
            ],
            "hybrid_full_and_tiles": True,
            "max_det_after_global_nms": FIXED_MAX_DET,
            "batch": args.batch,
            "device": args.device,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "rows": rows,
        "actual_wall_seconds": time.perf_counter() - total_started,
        "artifacts": {
            "csv": str(csv_path.resolve()),
            "markdown": str(markdown_path.resolve()),
            "details": str(details_path.resolve()),
            "report": str(report_path.resolve()),
        },
    }
    write_json_report(report_path, report)
    return report


def main() -> int:
    try:
        report = run(build_parser().parse_args())
    except Exception as exc:
        print(
            json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False, indent=2),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"status": report["status"], "rows": report["rows"], "artifacts": report["artifacts"]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
