#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from helmet_safety.training.ci_smoke import (  # noqa: E402
    reset_generated_output,
    run_one_epoch_smoke,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a self-contained one-epoch YOLO training gate"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "ci-smoke")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.force:
            raise FileExistsError(f"refusing to overwrite prior CI evidence: {output}")
        reset_generated_output(output, artifacts_root=ROOT / "artifacts")
    report = run_one_epoch_smoke(output)
    print(f"CI training smoke: {report['status']}")
    print(f"run: {report['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
