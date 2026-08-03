"""Characterize a device and run the exhaustive two-qubit Grover benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / "src"))

from q_error_id.haiqu_app import ExecutionConfig, HaiquErrorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Q-ErrorID raw diagnostics, reconstruct the local generator, "
            "and benchmark full learned error cancellation on all four "
            "two-qubit Grover targets across the selected three-edge subgraph."
        )
    )
    parser.add_argument("--device", default="fake_fez")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--validation-repeats",
        type=int,
        default=3,
        help="Independent repeats used only to enable or reject the full stack.",
    )
    parser.add_argument(
        "--evaluation-repeats",
        type=int,
        default=5,
        help="Held-out repeats used for reported means and 95% intervals.",
    )
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument(
        "--qubits",
        type=int,
        default=4,
        help="Size of the connected demo subgraph selected from the device.",
    )
    parser.add_argument("--response-regularization", type=float, default=3e-2)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=(2, 4, 8, 16),
        help="Even two-qubit depths for ideal-preserving CX-folding sweep.",
    )
    parser.add_argument(
        "--model",
        choices=("ridge", "qnn", "compare"),
        default="compare",
        help="Compare executes Ridge and all seven trained QNN circuits.",
    )
    parser.add_argument(
        "--require-cloud",
        action="store_true",
        help="Require authenticated Haiqu execution instead of local fallback.",
    )
    parser.add_argument("--no-local-fallback", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExecutionConfig.from_env(
        device=args.device,
        shots=args.shots,
        seed=args.seed,
        validation_repeats=args.validation_repeats,
        evaluation_repeats=args.evaluation_repeats,
        optimization_level=args.optimization_level,
        demo_qubit_count=args.qubits,
        response_regularization=args.response_regularization,
        benchmark_two_qubit_depths=tuple(args.depths),
        verbose=True,
        model_mode=args.model,
        require_cloud=args.require_cloud,
        allow_local_fallback=not args.no_local_fallback,
        output_root=args.output_root,
    )
    report = HaiquErrorPipeline(config).run(include_benchmark=True)
    with Path(report.artifacts["final_benchmark"]).open(
        newline="",
        encoding="utf-8",
    ) as stream:
        benchmark_rows = list(csv.DictReader(stream))
    response_payload = json.loads(
        Path(report.artifacts["algorithm_response_models"]).read_text(
            encoding="utf-8"
        )
    )
    print(
        json.dumps(
            {
                "execution_mode": report.execution_mode,
                "device_id": report.device_id,
                "physical_qubits": list(report.physical_qubits),
                "correction_improvement_tvd": (
                    report.correction_improvement_tvd
                ),
                "correction_validation": response_payload.get("validation"),
                "benchmark_summary": benchmark_rows,
                "artifacts": {
                    key: value
                    for key, value in report.artifacts.items()
                    if any(
                        token in key
                        for token in (
                            "benchmark",
                            "response",
                            "readout",
                            "validation",
                        )
                    )
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
