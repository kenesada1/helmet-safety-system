#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.service.registry import load_model_artifact  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify registered model, experiment, report, and SHA256 provenance"
    )
    parser.add_argument(
        "--registry", type=Path, default=ROOT / "configs" / "model_registry.json"
    )
    parser.add_argument("--model-id", default="e4-yolo11s-960-onnx")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifact = load_model_artifact(args.registry, args.model_id, verify=True)
    print(
        json.dumps(
            {
                "status": "verified",
                "model_id": artifact.model_id,
                "experiment": artifact.experiment,
                "backend": artifact.backend,
                "artifact": str(artifact.artifact_path),
                "report": str(artifact.report_path),
                "sha256": artifact.sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

