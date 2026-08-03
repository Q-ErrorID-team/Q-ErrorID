#!/usr/bin/env python
"""Run diagnostics, reconstruction, error-atlas, and mitigation comparisons."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / "src"))

from q_error_id.haiqu_app import ExecutionConfig, HaiquErrorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="fake_fez")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument("--response-regularization", type=float, default=3e-2)
    parser.add_argument(
        "--model",
        choices=("ridge", "qnn", "compare"),
        default="compare",
    )
    parser.add_argument("--require-cloud", action="store_true")
    parser.add_argument("--no-local-fallback", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExecutionConfig.from_env(
        device=args.device,
        shots=args.shots,
        seed=args.seed,
        optimization_level=args.optimization_level,
        response_regularization=args.response_regularization,
        verbose=True,
        model_mode=args.model,
        require_cloud=args.require_cloud,
        allow_local_fallback=not args.no_local_fallback,
        output_root=args.output_root,
    )
    report = HaiquErrorPipeline(config).run(include_benchmark=False)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
