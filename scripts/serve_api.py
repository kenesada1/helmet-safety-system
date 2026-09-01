#!/usr/bin/env python3
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the helmet inference HTTP API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-level", default="info")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be within [1, 65535]")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    import uvicorn

    uvicorn.run(
        "helmet_safety.service.api:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

