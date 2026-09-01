#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from helmet_safety.training.baseline import (  # noqa: E402
    allocate_experiment_name,
    format_evaluation_metrics,
    scan_training_log,
    write_json_report,
)
from helmet_safety.training.eval_common import (  # noqa: E402
    build_conclusions,
    build_val_kwargs,
    compare_with_m4_baseline,
)


class TeeStream:
    def __init__(self, terminal: object, log_file: object) -> None:
        self.terminal = terminal
        self.log_file = log_file

    def write(self, value: str) -> int:
        self.terminal.write(value)
        self.log_file.write(value)
        return len(value)

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", "utf-8") or "utf-8"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate M4.5 E1 best.pt on val only")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--run-name", default="m45_yolo11n_e100_640_val_001")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=None, help="Defaults to the actual training batch")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_val_plots(save_dir: Path) -> dict[str, str]:
    required = {
        "confusion_matrix": "confusion_matrix.png",
        "confusion_matrix_normalized": "confusion_matrix_normalized.png",
        "pr_curve": "BoxPR_curve.png",
        "f1_curve": "BoxF1_curve.png",
        "precision_curve": "BoxP_curve.png",
        "recall_curve": "BoxR_curve.png",
    }
    outputs = {key: (save_dir / filename).resolve() for key, filename in required.items()}
    missing = [path for path in outputs.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"missing or empty val plots: {missing}")
    return {key: str(path) for key, path in outputs.items()}


def render_markdown(report: dict[str, object]) -> str:
    metrics = report["val_evaluation"]["metrics"]
    comparison = report["comparison"]
    training = report["training"]
    conclusions = report["conclusions"]

    metric_rows = []
    for scope, source in (
        ("overall", metrics["overall"]),
        ("helmet", metrics["per_class"]["helmet"]),
        ("no_helmet", metrics["per_class"]["no_helmet"]),
    ):
        metric_rows.append(
            f"| {scope} | {source['precision']:.6f} | {source['recall']:.6f} | "
            f"{source['map50']:.6f} | {source['map50_95']:.6f} |"
        )

    comparison_rows = []
    for scope in ("overall", "helmet", "no_helmet"):
        for metric in ("precision", "recall", "map50", "map50_95"):
            item = comparison[scope][metric]
            comparison_rows.append(
                f"| {scope} | {metric} | {item['m4_baseline']:.6f} | {item['m45_e1']:.6f} | "
                f"{item['absolute_change']:+.6f} | {item['percentage_point_change']:+.4f} pp |"
            )

    analysis = training["training_analysis"]
    trailing = analysis["trailing_window"]
    losses = analysis["losses"]
    lines = [
        "# M4.5 控制实验 E1：100 epochs vs M4 baseline",
        "",
        "## 实验协议",
        "",
        f"- milestone={training['milestone']}；experiment_id={training['experiment_id']}；baseline_run={training['baseline_run']}",
        f"- 训练起点：`{training['pretrained_model']}`",
        f"- 实际训练：{analysis['epochs_completed']} epochs；best={analysis['best_epoch']}；last={analysis['last_epoch']}；early stopping={analysis['early_stopping_triggered']}",
        f"- 实际 batch={training['actual_batch']}；训练耗时={training['duration_seconds']:.2f} 秒",
        "- 本报告只评估 val；test 未评估，也未用于训练、选模或调参。",
        "",
        "## E1 best.pt 的 val 指标",
        "",
        "| 范围 | Precision | Recall | AP50 / mAP50 | AP50-95 / mAP50-95 |",
        "|---|---:|---:|---:|---:|",
        *metric_rows,
        "",
        f"- GPU inference：{metrics['speed_ms_per_image']['inference']:.4f} ms/image（{metrics['gpu_inference_images_per_second']:.2f} images/s）",
        f"- 全流水线平均耗时：{metrics['average_pipeline_ms_per_image']:.4f} ms/image",
        "",
        "## 与 M4 baseline 的 val 对比",
        "",
        "| 范围 | 指标 | M4 baseline | M4.5 E1 | 绝对变化 | 百分点变化 |",
        "|---|---|---:|---:|---:|---:|",
        *comparison_rows,
        "",
        "## Loss、平台期与过拟合",
        "",
        f"- train box：{losses['train']['box_loss']['first']:.6f} → {losses['train']['box_loss']['best']:.6f} → {losses['train']['box_loss']['last']:.6f}（first/best/last）",
        f"- train cls：{losses['train']['cls_loss']['first']:.6f} → {losses['train']['cls_loss']['best']:.6f} → {losses['train']['cls_loss']['last']:.6f}",
        f"- train dfl：{losses['train']['dfl_loss']['first']:.6f} → {losses['train']['dfl_loss']['best']:.6f} → {losses['train']['dfl_loss']['last']:.6f}",
        f"- val box：{losses['val']['box_loss']['first']:.6f} → {losses['val']['box_loss']['best']:.6f} → {losses['val']['box_loss']['last']:.6f}",
        f"- val cls：{losses['val']['cls_loss']['first']:.6f} → {losses['val']['cls_loss']['best']:.6f} → {losses['val']['cls_loss']['last']:.6f}",
        f"- val dfl：{losses['val']['dfl_loss']['first']:.6f} → {losses['val']['dfl_loss']['best']:.6f} → {losses['val']['dfl_loss']['last']:.6f}",
        f"- 最后 {trailing['epoch_count']} epochs：train 总 loss 变化 {trailing['train_total_loss']['change']:+.6f}，val 总 loss 变化 {trailing['val_total_loss']['change']:+.6f}，val mAP50-95 变化 {trailing['map50_95']['change']:+.6f}。",
        f"- post-50 状态：{conclusions['post_50_state']}；过拟合：{conclusions['overfitting_detected']}。",
        "",
        "## 结论",
        "",
        f"- overall Recall 提高：{conclusions['overall_recall_improved']}",
        f"- no_helmet Recall 提高：{conclusions['no_helmet_recall_improved']}",
        f"- no_helmet AP50-95 提高：{conclusions['no_helmet_map50_95_improved']}",
        f"- Precision 明显下降：{conclusions['precision_materially_declined']}",
        f"- best epoch 明显超过 50：{conclusions['best_epoch_beyond_50']}",
        f"- 后续继续使用 100 epochs 值得：{conclusions['epochs_100_worthwhile']}",
        "",
        "## 关键路径",
        "",
        f"- training report：`{report['paths']['training_report']}`",
        f"- best.pt：`{report['paths']['best_pt']}`",
        f"- last.pt：`{report['paths']['last_pt']}`",
        f"- training log：`{report['paths']['training_log']}`",
        f"- val evaluation dir：`{report['val_evaluation']['save_dir']}`",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from ultralytics import YOLO

    started_at = utc_now()
    started_clock = time.perf_counter()
    training_report_path = args.training_report.resolve()
    training = json.loads(training_report_path.read_text(encoding="utf-8"))
    if training.get("status") != "passed":
        raise ValueError(f"training did not pass: {training_report_path}")
    expected_metadata = {"milestone": "M4.5", "experiment_id": "E1", "baseline_run": "baseline_yolo11n_001"}
    if any(training.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"unexpected experiment metadata: {training_report_path}")
    if int(training["training_parameters"]["epochs"]) != 100:
        raise ValueError("E1 training report must request exactly 100 epochs")
    if args.imgsz != 640:
        raise ValueError("M4.5 E1 val evaluation requires imgsz=640")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.device!r} requested but unavailable")

    best_pt = Path(training["outputs"]["best_pt"]).resolve()
    last_pt = Path(training["outputs"]["last_pt"]).resolve()
    if not best_pt.is_file() or not last_pt.is_file():
        raise FileNotFoundError("best.pt or last.pt is missing")
    evaluation_dir = args.artifacts_dir.resolve() / "evaluation"
    logs_dir = args.artifacts_dir.resolve() / "logs"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_name = allocate_experiment_name(evaluation_dir, logs_dir, args.run_name)
    log_path = logs_dir / f"{run_name}.log"
    batch = int(args.batch or training["actual_batch"])

    with log_path.open("x", encoding="utf-8", buffering=1) as log_handle:
        with redirect_stdout(TeeStream(sys.stdout, log_handle)), redirect_stderr(TeeStream(sys.stderr, log_handle)):
            print(json.dumps({"split": "val", "training_report": str(training_report_path)}, indent=2))
            best_model = YOLO(str(best_pt), task="detect")
            last_model = YOLO(str(last_pt), task="detect")
            expected_names = {0: "helmet", 1: "no_helmet"}
            if dict(best_model.names) != expected_names or dict(last_model.names) != expected_names:
                raise ValueError(f"unexpected class mapping: best={best_model.names}, last={last_model.names}")
            val_kwargs = build_val_kwargs(
                data_yaml=Path(training["data"]["dataset_yaml"]),
                project_dir=evaluation_dir,
                run_name=run_name,
                imgsz=args.imgsz,
                batch=batch,
                workers=args.workers,
                device=args.device,
                seed=args.seed,
            )
            metrics_object = best_model.val(**val_kwargs)
            save_dir = Path(metrics_object.save_dir).resolve()
            metrics = format_evaluation_metrics(
                metrics_object.results_dict,
                metrics_object.summary(normalize=True, decimals=10),
                metrics_object.speed,
            )

    values = [
        value
        for scope in (metrics["overall"], *metrics["per_class"].values())
        for key, value in scope.items()
        if key not in {"images", "instances"}
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("val evaluation produced non-finite metrics")
    plots = validate_val_plots(save_dir)
    warning_audit = scan_training_log(log_path)
    if any(warning_audit[key] for key in ("jpeg_auto_repair_warning", "cache_version_warning", "corrupt_data_warning")):
        raise RuntimeError(f"invalidating data/cache warning found in {log_path}")
    comparison = compare_with_m4_baseline(metrics)
    conclusions = build_conclusions(comparison, training["training_analysis"])
    report: dict[str, object] = {
        "status": "passed",
        **expected_metadata,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "training": training,
        "val_evaluation": {
            "split": "val",
            "save_dir": str(save_dir),
            "parameters": val_kwargs,
            "metrics": metrics,
            "plots": plots,
            "warning_audit": warning_audit,
        },
        "comparison": comparison,
        "conclusions": conclusions,
        "test_used_for_training_or_selection": False,
        "test_evaluated": False,
        "paths": {
            "training_report": str(training_report_path),
            "best_pt": str(best_pt),
            "last_pt": str(last_pt),
            "training_log": training["console_log"],
            "evaluation_log": str(log_path.resolve()),
        },
    }
    report_json = save_dir / "m45_e1_val_report.json"
    comparison_json = save_dir / "e1_vs_baseline.json"
    comparison_md = save_dir / "e1_vs_baseline.md"
    report["paths"].update(
        {
            "evaluation_report": str(report_json.resolve()),
            "comparison_json": str(comparison_json.resolve()),
            "comparison_markdown": str(comparison_md.resolve()),
        }
    )
    write_json_report(report_json, report)
    write_json_report(
        comparison_json,
        {"milestone": "M4.5", "experiment_id": "E1", "comparison": comparison, "conclusions": conclusions},
    )
    comparison_md.write_text(render_markdown(report), encoding="utf-8")

    training.update(
        {
            "best_epoch": training["training_analysis"]["best_epoch"],
            "last_epoch": training["training_analysis"]["last_epoch"],
            "actual_epochs": training["training_analysis"]["epochs_completed"],
            "early_stopping": training["training_analysis"]["early_stopping_triggered"],
            "val_evaluation": report["val_evaluation"],
            "val_overall_metrics": metrics["overall"],
            "val_per_class_metrics": metrics["per_class"],
            "loss_analysis": training["training_analysis"],
            "overfitting_judgment": training["training_analysis"]["overfitting"],
            "baseline_comparison": comparison,
            "experiment_conclusions": conclusions,
            "val_evaluation_dir": str(save_dir),
            "test_used_for_training_or_selection": False,
            "test_evaluated": False,
        }
    )
    write_json_report(training_report_path, training, overwrite=True)
    print(json.dumps({"status": "passed", "val_dir": str(save_dir), "report": str(report_json)}, indent=2))
    return report


def main() -> int:
    try:
        run(build_parser().parse_args())
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
