#!/usr/bin/env python3
"""Generate Q-ErrorID datasets, identifiability reports, and validation data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_error_id.core import (
    analyze_identifiability,
    build_channel,
    one_qubit_protocol,
    representative_parameters,
    select_protocol,
    two_qubit_candidate_protocol,
    validate_channel,
)
from q_error_id.core.datasets import (
    FAMILIES,
    generate_dataset_split,
    sample_parameters,
    save_dataset,
    sha256_file,
)
from q_error_id.core.protocols import ReadoutConfusion

RECOMMENDED_SIZES = {
    "1q_mixed_channel": {"train": 6000, "validation": 1000, "test": 1000},
    "2q_mixed_channel": {"train": 10000, "validation": 1500, "test": 1500},
}
DEMO_SIZES = {
    "1q_mixed_channel": {"train": 256, "validation": 64, "test": 64},
    "2q_mixed_channel": {"train": 192, "validation": 48, "test": 48},
}
SMOKE_SIZES = {
    "1q_mixed_channel": {"train": 24, "validation": 8, "test": 8},
    "2q_mixed_channel": {"train": 16, "validation": 6, "test": 6},
}


def serialize_manifest_path(path: Path) -> str:
    """Return a project-relative path when possible, otherwise an absolute one."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("recommended", "demo", "smoke"),
        default="recommended",
        help="Dataset sizes. Generated deliverables use demo to keep runtime modest.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "datasets",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "core",
    )
    parser.add_argument("--readout-plus-to-minus", type=float, default=0.018)
    parser.add_argument("--readout-minus-to-plus", type=float, default=0.024)
    return parser.parse_args()


def write_identifiability_report(destination: Path) -> dict[str, object]:
    """Design the two-qubit bank and serialize both protocol reports."""

    one_parameters = representative_parameters("1Q")
    one_protocol = one_qubit_protocol()
    one_report = analyze_identifiability(one_parameters, one_protocol)

    two_parameters = representative_parameters("CX", basis=("ZI", "IZ", "ZX", "ZZ"))
    two_protocol, selection_report = select_protocol(
        two_parameters,
        two_qubit_candidate_protocol(),
        target_features=80,
    )
    two_report = analyze_identifiability(two_parameters, two_protocol)
    two_gate_only_report = analyze_identifiability(
        two_parameters,
        two_protocol,
        include_kappa=False,
    )
    report = {
        "finite_difference_point": {
            "one_qubit": one_parameters.as_vector().tolist(),
            "two_qubit": two_parameters.as_vector().tolist(),
        },
        "one_qubit": one_report.to_dict(),
        "two_qubit": two_report.to_dict(),
        "two_qubit_gate_specific": two_gate_only_report.to_dict(),
        "two_qubit_selection": selection_report,
        "interpretation": (
            "The minimal reliable subset is a local rank-revealing QR subset. "
            "Remaining selected features are redundant for noiseless local "
            "identifiability but improve finite-shot robustness. The full "
            "two-qubit report includes two local damping rates; the "
            "two_qubit_gate_specific report treats those previously estimated "
            "local rates as fixed and therefore has eight parameters."
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def write_validation_csv(destination: Path, seed: int) -> None:
    """Validate a deterministic sample of both channel families."""

    rng = np.random.default_rng(seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "sample",
        "complete_positive",
        "trace_preserving",
        "hermiticity_preserving",
        "minimum_choi_eigenvalue",
        "trace_preservation_violation",
        "hermiticity_violation",
    ]
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for family in FAMILIES:
            for sample_index in range(8):
                channel = build_channel(sample_parameters(family, rng))
                values = validate_channel(channel)
                writer.writerow({"family": family, "sample": sample_index, **values})


def main() -> None:
    args = parse_args()
    sizes = {
        "recommended": RECOMMENDED_SIZES,
        "demo": DEMO_SIZES,
        "smoke": SMOKE_SIZES,
    }[args.profile]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion = ReadoutConfusion(
        p_plus_to_minus=args.readout_plus_to_minus,
        p_minus_to_plus=args.readout_minus_to_plus,
    )
    files: list[dict[str, object]] = []
    family_offsets = {"1q_mixed_channel": 0, "2q_mixed_channel": 100_000}
    split_offsets = {"train": 0, "validation": 10_000, "test": 20_000}

    for family in FAMILIES:
        for split, size in sizes[family].items():
            seed = args.seed + family_offsets[family] + split_offsets[split]
            arrays = generate_dataset_split(
                family,
                size,
                seed=seed,
                readout_confusion=confusion,
            )
            prefix = "1q" if family.startswith("1q") else "2q"
            path = args.output_dir / f"{prefix}_mixed_channel_{split}.npz"
            save_dataset(path, arrays)
            files.append(
                {
                    "family": family,
                    "split": split,
                    "size": size,
                    "seed": seed,
                    "path": serialize_manifest_path(path),
                    "sha256": sha256_file(path),
                    "feature_count": int(arrays["X_exact"].shape[1]),
                }
            )
            print(f"wrote {path} ({size} samples)")

    identifiability_path = args.results_dir / "identifiability_report.json"
    validation_path = args.results_dir / "channel_validation.csv"
    report = write_identifiability_report(identifiability_path)
    write_validation_csv(validation_path, args.seed + 999_999)
    manifest = {
        "schema_version": "1.0",
        "generator": "scripts/generate_datasets.py",
        "profile": args.profile,
        "base_seed": args.seed,
        "readout_confusion": confusion.to_dict(),
        "files": files,
        "identifiability_report": serialize_manifest_path(identifiability_path),
        "channel_validation": serialize_manifest_path(validation_path),
        "one_qubit_jacobian_rank": report["one_qubit"]["rank"],
        "two_qubit_jacobian_rank": report["two_qubit"]["rank"],
        "two_qubit_gate_specific_jacobian_rank": report["two_qubit_gate_specific"][
            "rank"
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
