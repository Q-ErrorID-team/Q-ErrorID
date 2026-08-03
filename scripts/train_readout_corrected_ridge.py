"""Train deployment Ridge models on readout-debiased finite-shot features."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / "src"))

from q_error_id.haiqu_app.models import NumpyLinearModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Ridge deployment artifacts on finite-shot features after "
            "removing the synthetic assignment-channel bias."
        )
    )
    parser.add_argument(
        "--shots",
        nargs="+",
        type=int,
        default=[256, 1024, 4096, 8192],
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "artifacts" / "datasets",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "artifacts" / "models",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = []
    for family in ("1q", "2q"):
        dataset = args.dataset_root / f"{family}_mixed_channel_train.npz"
        if not dataset.exists():
            raise FileNotFoundError(dataset)
        for shots in args.shots:
            model = NumpyLinearModel.fit_dataset(
                dataset,
                family=family,
                shots=shots,
                readout_corrected=True,
            )
            path = args.model_root / (
                f"ridge_{family}_{shots}_readout_corrected.npz"
            )
            model.save(path)
            outputs.append(
                {
                    "family": family,
                    "shots": shots,
                    "path": path.as_posix(),
                    "training_features": model.source,
                }
            )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
