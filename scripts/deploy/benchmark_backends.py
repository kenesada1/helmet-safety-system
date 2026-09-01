from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PT = ROOT / "artifacts/training/m45_yolo11s_e75_960_001/weights/best.pt"
DEFAULT_ONNX = ROOT / "artifacts/deployment/e4_yolo11s_960.onnx"
DEFAULT_JSON = ROOT / "artifacts/deployment/benchmark_report.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _linear_percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latencies_ms(latencies_ms: list[float]) -> dict[str, int | float]:
    if not latencies_ms:
        raise ValueError("at least one latency is required")
    if any(not math.isfinite(value) or value <= 0 for value in latencies_ms):
        raise ValueError("latencies must be finite positive milliseconds")
    mean_ms = statistics.fmean(latencies_ms)
    return {
        "iterations": len(latencies_ms),
        "mean_ms": round(mean_ms, 6),
        "p50_ms": round(_linear_percentile(latencies_ms, 0.50), 6),
        "p95_ms": round(_linear_percentile(latencies_ms, 0.95), 6),
        "min_ms": round(min(latencies_ms), 6),
        "max_ms": round(max(latencies_ms), 6),
        "fps": round(1000.0 / mean_ms, 6),
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    config = report["config"]
    environment = report["environment"]
    lines = [
        "# E4 ONNX 性能测试",
        "",
        f"测试配置：batch={config['batch']}、imgsz={config['imgsz']}、warmup={config['warmup']}、正式测试 {config['iterations']} 次。输入图片已预先以 OpenCV BGR 格式加载到内存，延迟不包含文件读取。",
        "",
        "| 后端 | 设备 | 平均延迟 | P50 | P95 | FPS |",
        "| -- | -- | ---: | --: | --: | --: |",
    ]
    notes: list[str] = []
    for backend in report["backends"]:
        if backend["status"] == "ok":
            lines.append(
                f"| {backend['backend']} | {backend['device']} | {backend['mean_ms']:.2f} ms | "
                f"{backend['p50_ms']:.2f} ms | {backend['p95_ms']:.2f} ms | {backend['fps']:.2f} |"
            )
        else:
            lines.append(
                f"| {backend['backend']} | {backend.get('device', 'unavailable')} | unavailable | unavailable | unavailable | unavailable |"
            )
            notes.append(f"- {backend['backend']}：{backend['status']} — {backend.get('reason', '未提供原因')}")
    lines.extend(
        [
            "",
            "## 测试环境",
            "",
            f"- GPU：{environment.get('gpu', 'unavailable')}",
            f"- CPU：{environment.get('cpu', 'unknown')}",
            f"- 操作系统：{environment.get('os', 'unknown')}",
            f"- Python：{environment.get('python', 'unknown')}",
            f"- PyTorch：{environment.get('torch', 'unknown')}",
            f"- Ultralytics：{environment.get('ultralytics', 'unknown')}",
            f"- ONNX Runtime：{environment.get('onnxruntime', 'unknown')}",
        ]
    )
    if notes:
        lines.extend(["", "## 未执行项", "", *notes])
    provider_lines = [
        f"- {item['backend']}：{', '.join(item.get('provider', []))}"
        for item in report["backends"]
        if item.get("provider")
    ]
    if provider_lines:
        lines.extend(["", "## 实际 ONNX Runtime Provider", "", *provider_lines])
    lines.extend(
        [
            "",
            "> 以上结果是当前机器上的 batch=1、imgsz=960 端到端模型调用耗时（预处理、推理和 NMS），不包含磁盘读取；不是 val/test 精度指标。",
            "",
        ]
    )
    return "\n".join(lines)


def _load_images(source: Path) -> tuple[list[np.ndarray], list[str]]:
    source = source.expanduser().resolve()
    if source.is_file():
        paths = [source]
    elif source.is_dir():
        paths = sorted(path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(f"benchmark source does not exist: {source}")
    if not paths:
        raise FileNotFoundError(f"no supported images found in: {source}")
    images: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read benchmark image: {path}")
        images.append(image)
    return images, [str(path) for path in paths]


def _active_gpu_compute_processes() -> tuple[list[dict[str, str]], str | None]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return [], f"could not verify GPU process state with nvidia-smi: {exc}"
    processes: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        if len(parts) == 3 and parts[0] != str(os.getpid()):
            processes.append({"pid": parts[0], "process_name": parts[1], "memory_mib": parts[2]})
    return processes, None


def gpu_skip_reason(
    *,
    gpu_available: bool,
    gpu_processes: list[dict[str, str]],
    gpu_check_error: str | None,
    allow_busy_gpu: bool,
) -> str | None:
    if not gpu_available:
        return "CUDA is unavailable"
    if allow_busy_gpu:
        return None
    if gpu_check_error:
        return gpu_check_error
    if gpu_processes:
        return "formal GPU benchmark skipped because another GPU compute process is active"
    return None


def _ort_providers_from_yolo(model: object) -> list[str]:
    predictor = getattr(model, "predictor", None)
    backend = getattr(predictor, "model", None)
    session = getattr(backend, "session", None)
    if session is None or not hasattr(session, "get_providers"):
        return []
    return [str(value) for value in session.get_providers()]


def _benchmark_yolo(
    *,
    backend_name: str,
    weights: Path,
    device: str,
    images: list[np.ndarray],
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    warmup: int,
    iterations: int,
    synchronize: Callable[[], None] | None,
) -> tuple[dict[str, Any], object]:
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def predict(index: int) -> object:
        return model.predict(
            source=images[index % len(images)],
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            batch=1,
            verbose=False,
        )[0]

    for index in range(warmup):
        predict(index)
    if synchronize is not None:
        synchronize()
    latencies_ms: list[float] = []
    last_result: object | None = None
    for index in range(iterations):
        if synchronize is not None:
            synchronize()
        started = perf_counter()
        last_result = predict(index)
        if synchronize is not None:
            synchronize()
        latencies_ms.append((perf_counter() - started) * 1000.0)
    summary: dict[str, Any] = {
        "backend": backend_name,
        "status": "ok",
        "device": device,
        **summarize_latencies_ms(latencies_ms),
    }
    boxes = getattr(last_result, "boxes", None)
    summary["last_detection_count"] = len(boxes) if boxes is not None else 0
    return summary, model


def run_benchmarks(
    *,
    pt_weights: Path,
    onnx_weights: Path,
    source: Path,
    output_json: Path,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
    warmup: int,
    iterations: int,
    force: bool,
    allow_busy_gpu: bool = False,
) -> dict[str, Any]:
    if imgsz < 1 or max_det < 1 or warmup < 0:
        raise ValueError("imgsz/max_det must be positive and warmup must be non-negative")
    if iterations < 100:
        raise ValueError("formal benchmark requires at least 100 iterations")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf and iou must be within [0, 1]")
    pt_weights = pt_weights.expanduser().resolve()
    onnx_weights = onnx_weights.expanduser().resolve()
    if not pt_weights.is_file():
        raise FileNotFoundError(f"PT weights do not exist: {pt_weights}")
    if not onnx_weights.is_file():
        raise FileNotFoundError(f"ONNX weights do not exist: {onnx_weights}")
    output_json = output_json.expanduser().resolve()
    output_md = output_json.with_suffix(".md")
    existing = [path for path in (output_json, output_md) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite benchmark reports without --force: "
            + ", ".join(str(path) for path in existing)
        )
    images, image_paths = _load_images(source)

    import onnxruntime as ort
    import torch
    import ultralytics

    available_providers = list(ort.get_available_providers())
    gpu_processes, gpu_check_error = _active_gpu_compute_processes()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    environment = {
        "gpu": gpu_name,
        "cpu": platform.processor() or platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "ultralytics": ultralytics.__version__,
        "onnxruntime": ort.__version__,
        "onnxruntime_available_providers": available_providers,
        "gpu_compute_processes_before_benchmark": gpu_processes,
        "gpu_process_check_error": gpu_check_error,
        "gpu_busy_override": allow_busy_gpu,
    }
    results: list[dict[str, Any]] = []
    gpu_busy_reason = gpu_skip_reason(
        gpu_available=torch.cuda.is_available(),
        gpu_processes=gpu_processes,
        gpu_check_error=gpu_check_error,
        allow_busy_gpu=allow_busy_gpu,
    )

    if not torch.cuda.is_available():
        results.append(
            {"backend": "PyTorch CUDA", "status": "unavailable", "device": "unavailable", "reason": "torch.cuda.is_available() is false"}
        )
    elif gpu_busy_reason:
        results.append(
            {"backend": "PyTorch CUDA", "status": "skipped", "device": "cuda:0", "reason": gpu_busy_reason, "active_gpu_processes": gpu_processes}
        )
    else:
        result, _ = _benchmark_yolo(
            backend_name="PyTorch CUDA",
            weights=pt_weights,
            device="0",
            images=images,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            warmup=warmup,
            iterations=iterations,
            synchronize=torch.cuda.synchronize,
        )
        result["device"] = f"cuda:0 ({gpu_name})"
        results.append(result)

    if "CUDAExecutionProvider" not in available_providers:
        results.append(
            {
                "backend": "ONNX Runtime CUDA",
                "status": "unavailable",
                "device": "unavailable",
                "provider": available_providers,
                "reason": "CUDAExecutionProvider is not installed or unavailable",
            }
        )
    elif gpu_busy_reason:
        results.append(
            {
                "backend": "ONNX Runtime CUDA",
                "status": "skipped",
                "device": "cuda:0",
                "provider": [],
                "available_providers": available_providers,
                "reason": gpu_busy_reason,
                "active_gpu_processes": gpu_processes,
            }
        )
    else:
        result, model = _benchmark_yolo(
            backend_name="ONNX Runtime CUDA",
            weights=onnx_weights,
            device="0",
            images=images,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            warmup=warmup,
            iterations=iterations,
            synchronize=torch.cuda.synchronize,
        )
        result["provider"] = _ort_providers_from_yolo(model)
        if "CUDAExecutionProvider" not in result["provider"]:
            result["status"] = "fallback"
            result["reason"] = "CUDA was requested but the active ONNX Runtime session fell back to CPU"
        results.append(result)

    cpu_result, cpu_model = _benchmark_yolo(
        backend_name="ONNX Runtime CPU",
        weights=onnx_weights,
        device="cpu",
        images=images,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        warmup=warmup,
        iterations=iterations,
        synchronize=None,
    )
    cpu_result["device"] = "CPU"
    cpu_result["provider"] = _ort_providers_from_yolo(cpu_model)
    if "CPUExecutionProvider" not in cpu_result["provider"]:
        raise RuntimeError(f"ONNX CPU benchmark did not use CPUExecutionProvider: {cpu_result['provider']}")
    results.append(cpu_result)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": {
            "batch": 1,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "warmup": warmup,
            "iterations": iterations,
            "timing_scope": "in-memory BGR preprocessing + inference + NMS; excludes file I/O",
        },
        "inputs": {"source": str(source.expanduser().resolve()), "images": image_paths, "loaded_image_count": len(images)},
        "models": {"pt": str(pt_weights), "onnx": str(onnx_weights)},
        "environment": environment,
        "backends": results,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown_report(report), encoding="utf-8")
    print(render_markdown_report(report))
    print(f"JSON report: {output_json}")
    print(f"Markdown report: {output_md}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark E4 PyTorch and ONNX Runtime backends")
    parser.add_argument("--pt-weights", type=Path, default=DEFAULT_PT)
    parser.add_argument("--onnx-weights", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--source", type=Path, default=Path(r"D:\datasets\SHWD\processed\images\val\000000.jpg"))
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="run GPU benchmarks despite detected GPU processes; use only after confirming they are paused",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_benchmarks(
        pt_weights=args.pt_weights,
        onnx_weights=args.onnx_weights,
        source=args.source,
        output_json=args.output,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        warmup=args.warmup,
        iterations=args.iterations,
        force=args.force,
        allow_busy_gpu=args.allow_busy_gpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
