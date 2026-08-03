#!/usr/bin/env python3
"""Train Q-ErrorID QNNs and classical baselines, then build all deliverables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/q_error_id_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q_error_id.core.datasets import (
    generate_dataset_split,
    save_dataset,
)
from q_error_id.core.protocols import ReadoutConfusion
from q_error_id.models import (
    HybridQNN,
    LossConfig,
    PhysicalBaseline,
    QNNTrainingConfig,
    comparable_hidden_units,
    evaluate_model,
    export_qiskit_inference,
    load_model_dataset,
    randomized_shot_augment,
)

DEMO_SIZES = {
    "1q": {"train": 256, "validation": 64, "test": 64},
    "2q": {"train": 192, "validation": 48, "test": 48},
}
SMOKE_SIZES = {
    "1q": {"train": 32, "validation": 12, "test": 12},
    "2q": {"train": 24, "validation": 10, "test": 10},
}
FAMILY_NAMES = {"1q": "1q_mixed_channel", "2q": "2q_mixed_channel"}
SHOT_KEYS = {
    "exact": "X_exact",
    "4096-shot": "X_shot_4096",
    "1024-shot": "X_shot_1024",
    "256-shot": "X_shot_256",
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
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "datasets",
    )
    parser.add_argument("--profile", choices=("demo", "smoke"), default="demo")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument(
        "--skip-data-generation",
        action="store_true",
        help="Fail if an Agent-1 split is missing.",
    )
    return parser.parse_args()


def ensure_datasets(
    dataset_dir: Path,
    profile: str,
    seed: int,
    *,
    allow_generation: bool,
) -> None:
    """Use Agent 1's generator contract when prebuilt artifacts are absent."""

    sizes = DEMO_SIZES if profile == "demo" else SMOKE_SIZES
    confusion = ReadoutConfusion(
        p_plus_to_minus=0.018,
        p_minus_to_plus=0.024,
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)
    seed_offsets = {"1q": 0, "2q": 100_000}
    split_offsets = {"train": 0, "validation": 10_000, "test": 20_000}
    manifest_files: list[dict[str, Any]] = []
    for family in ("1q", "2q"):
        for split, size in sizes[family].items():
            path = dataset_dir / f"{family}_mixed_channel_{split}.npz"
            needs_generation = not path.exists()
            if path.exists():
                needs_generation = load_model_dataset(path).size != size
            if needs_generation:
                if not allow_generation:
                    raise FileNotFoundError(
                        f"Agent-1 dataset is missing or has the wrong profile: {path}"
                    )
                arrays = generate_dataset_split(
                    FAMILY_NAMES[family],
                    size,
                    seed=seed + seed_offsets[family] + split_offsets[split],
                    readout_confusion=confusion,
                )
                save_dataset(path, arrays)
                print(f"generated {path} ({size} samples)")
            loaded = load_model_dataset(path)
            manifest_files.append(
                {
                    "family": family,
                    "split": split,
                    "path": serialize_manifest_path(path),
                    "size": loaded.size,
                    "feature_count": loaded.X.shape[1],
                }
            )
    manifest = {
        "schema_version": "agent1-compatible-1.0",
        "profile": profile,
        "files": manifest_files,
        "generator_contract": "q_error_id.core.datasets.generate_dataset_split",
    }
    (dataset_dir / "agent2_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def load_family(dataset_dir: Path, family: str):
    return {
        split: load_model_dataset(dataset_dir / f"{family}_mixed_channel_{split}.npz")
        for split in ("train", "validation", "test")
    }


def train_family(
    family: str,
    data: dict[str, Any],
    args: argparse.Namespace,
    model_dir: Path,
    circuit_dir: Path,
) -> dict[str, Any]:
    train = data["train"]
    validation = data["validation"]
    test = data["test"]
    # Both model families fit the four-qubit demonstration subgraph. The 2Q
    # target still has eight outputs because four Z and four nearest-neighbour
    # ZZ observables provide an eight-dimensional learned quantum readout.
    n_qubits = 4
    batch_size = min(args.batch_size, train.size)
    base_training = {
        "epochs": args.epochs,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "gradient_clip": 2.0,
        "seed": args.seed + (0 if family == "1q" else 1000),
    }
    loss = LossConfig(
        w_alpha=1.0,
        w_gamma=1.0,
        w_kappa=1.0,
        lambda_sparse=1e-4,
        lambda_classification=0.03,
    )

    exact_qnn = HybridQNN(
        train.X.shape[1],
        family,
        n_qubits=n_qubits,
        n_layers=2,
        entanglement="ring",
        auxiliary_classifier=True,
        seed=args.seed + (11 if family == "1q" else 1011),
    )
    print(f"training exact QNN {family}")
    exact_qnn.fit(
        train.X,
        train.y,
        validation.X,
        validation.y,
        config=QNNTrainingConfig(**base_training, shot_robust=False),
        loss_config=loss,
    )

    robust_qnn = HybridQNN(
        train.X.shape[1],
        family,
        n_qubits=n_qubits,
        n_layers=2,
        entanglement="ring",
        auxiliary_classifier=True,
        seed=args.seed + (29 if family == "1q" else 1029),
    )
    robust_train = train.feature_variants.get("X_readout_exact", train.X)
    robust_validation = validation.feature_variants.get("X_shot_1024", validation.X)
    print(f"training shot-robust QNN {family}")
    robust_qnn.fit(
        robust_train,
        train.y,
        robust_validation,
        validation.y,
        config=QNNTrainingConfig(**base_training, shot_robust=True),
        loss_config=loss,
    )

    hidden = comparable_hidden_units(
        train.X.shape[1],
        train.spec.n_outputs,
        exact_qnn.trainable_parameter_count,
    )
    ridge = PhysicalBaseline(family, "ridge", seed=args.seed).fit(train.X, train.y)
    deployment_ridges = {
        shots: PhysicalBaseline(
            family,
            "ridge",
            seed=args.seed + shots,
        ).fit(
            train.feature_variants[f"X_shot_{shots}"],
            train.y,
        )
        for shots in (256, 1024, 4096, 8192)
    }
    print(f"training comparable MLP {family} (hidden={hidden})")
    mlp = PhysicalBaseline(family, "mlp", hidden_units=hidden, seed=args.seed).fit(
        train.X, train.y
    )

    rng = np.random.default_rng(args.seed + (77 if family == "1q" else 1077))
    robust_copies = 3
    repeated_features = np.tile(robust_train, (robust_copies, 1))
    augmented = randomized_shot_augment(repeated_features, rng)
    robust_mlp = PhysicalBaseline(
        family, "mlp", hidden_units=hidden, seed=args.seed + 1
    ).fit(
        np.concatenate([train.X, augmented], axis=0),
        np.tile(train.y, (robust_copies + 1, 1)),
    )

    paths = {
        "qnn_exact": exact_qnn.save(model_dir / f"qnn_{family}_exact.pt"),
        "qnn_shot_robust": robust_qnn.save(model_dir / f"qnn_{family}_shot_robust.pt"),
        "qnn_primary": exact_qnn.save(model_dir / f"qnn_{family}.pt"),
        "ridge": ridge.save(model_dir / f"ridge_{family}.pt"),
        "mlp": mlp.save(model_dir / f"mlp_{family}.pt"),
        "mlp_shot_robust": robust_mlp.save(model_dir / f"mlp_{family}_shot_robust.pt"),
        **{
            f"ridge_deployment_{shots}": model.save(
                model_dir / f"ridge_{family}_{shots}.npz"
            )
            for shots, model in deployment_ridges.items()
        },
    }
    circuit_paths = export_qiskit_inference(
        exact_qnn,
        circuit_dir / f"qnn_{family}_inference",
        reference_features=test.X[0],
    )

    full_models = {
        "QNN": exact_qnn,
        "MLP": mlp,
        "Ridge": ridge,
    }
    evaluations = {}
    for name, model in full_models.items():
        print(f"evaluating {name} {family}")
        evaluations[name] = evaluate_model(
            model,
            test.X,
            test.y,
            test.known_kappa,
            test.spec,
            metadata=test.metadata,
        )

    robustness_models = {
        "QNN-exact": exact_qnn,
        "QNN-shot-robust": robust_qnn,
        "MLP-exact": mlp,
        "MLP-shot-robust": robust_mlp,
    }
    robustness_rows = []
    for model_name, model in robustness_models.items():
        for regime, key in SHOT_KEYS.items():
            regime_features = test.feature_variants[key]
            prediction = model.predict(regime_features)
            robustness_rows.append(
                {
                    "family": family,
                    "model": model_name,
                    "feature_regime": regime,
                    "parameter_mae": float(np.mean(np.abs(prediction - test.y))),
                    "parameter_rmse": float(
                        np.sqrt(np.mean((prediction - test.y) ** 2))
                    ),
                }
            )

    return {
        "family": family,
        "data": data,
        "models": {
            **full_models,
            "QNN-shot-robust": robust_qnn,
            "MLP-shot-robust": robust_mlp,
        },
        "evaluations": evaluations,
        "robustness_rows": robustness_rows,
        "paths": {key: str(path) for key, path in paths.items()},
        "circuits": {key: str(path) for key, path in circuit_paths.items()},
        "hidden_units": hidden,
        "validation_parameter_mae": {
            "QNN": float(
                np.mean(np.abs(exact_qnn.predict(validation.X) - validation.y))
            ),
            "QNN-shot-robust@1024": float(
                np.mean(
                    np.abs(
                        robust_qnn.predict(validation.feature_variants["X_shot_1024"])
                        - validation.y
                    )
                )
            ),
            "MLP": float(np.mean(np.abs(mlp.predict(validation.X) - validation.y))),
            "Ridge": float(np.mean(np.abs(ridge.predict(validation.X) - validation.y))),
        },
    }


def write_predictions(result: dict[str, Any], predictions_dir: Path) -> None:
    family = result["family"]
    test = result["data"]["test"]
    evaluation = result["evaluations"]["QNN"]
    frame: dict[str, Any] = {"sample": np.arange(test.size)}
    for index, name in enumerate(test.spec.names):
        frame[f"true_{name}"] = test.y[:, index]
        frame[f"predicted_{name}"] = evaluation.predictions[:, index]
    details = {row["sample"]: row for row in evaluation.per_sample}
    for key in (
        "choi_fidelity",
        "ptm_error",
        "heldout_expectation_rmse",
        "fidelity_before_compensation",
        "fidelity_after_compensation",
        "remaining_stochastic_error",
        "strategy",
    ):
        frame[key] = [details[index][key] for index in range(test.size)]
    pd.DataFrame(frame).to_csv(predictions_dir / f"test_{family}.csv", index=False)


def comparison_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        exact_256 = {
            row["model"]: row["parameter_mae"]
            for row in result["robustness_rows"]
            if row["feature_regime"] == "256-shot"
        }
        for name in ("QNN", "MLP", "Ridge"):
            model = result["models"][name]
            metrics = result["evaluations"][name].summary
            robustness_key = f"{name}-exact" if name != "Ridge" else None
            rows.append(
                {
                    "family": result["family"],
                    "model": name,
                    "trainable_parameters": model.trainable_parameter_count,
                    "training_time_seconds": model.training_time,
                    "inference_time_seconds_per_sample": metrics[
                        "inference_seconds_per_sample"
                    ],
                    "parameter_mae": metrics["parameter_mae"],
                    "choi_fidelity": metrics["choi_fidelity"],
                    "ptm_error": metrics["ptm_error"],
                    "shot_robustness_256_mae": (
                        exact_256.get(robustness_key, np.nan)
                        if robustness_key
                        else np.nan
                    ),
                }
            )
    return rows


def plot_true_vs_predicted(result: dict[str, Any], results_dir: Path) -> None:
    test = result["data"]["test"]
    prediction = result["evaluations"]["QNN"].predictions
    count = test.spec.n_outputs
    columns = 4
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False
    )
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.axis("off")
            continue
        truth = test.y[:, index]
        predicted = prediction[:, index]
        lower = min(float(truth.min()), float(predicted.min()))
        upper = max(float(truth.max()), float(predicted.max()))
        axis.scatter(truth, predicted, s=18, alpha=0.75)
        axis.plot([lower, upper], [lower, upper], "k--", linewidth=1)
        axis.set_title(test.spec.names[index])
        axis.set_xlabel("true")
        axis.set_ylabel("predicted")
    figure.suptitle(f"{result['family']} QNN: true vs predicted")
    figure.tight_layout()
    figure.savefig(
        results_dir / f"{result['family']}_true_vs_predicted.png",
        dpi=160,
    )
    plt.close(figure)


def plot_choi_histograms(results: list[dict[str, Any]], results_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, result in zip(axes, results):
        for name, color in (("QNN", "tab:blue"), ("MLP", "tab:orange")):
            values = [
                row["choi_fidelity"] for row in result["evaluations"][name].per_sample
            ]
            axis.hist(values, bins=12, alpha=0.55, label=name, color=color)
        axis.set_title(result["family"])
        axis.set_xlabel("Choi-state fidelity")
        axis.set_ylabel("count")
        axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "choi_fidelity_histograms.png", dpi=160)
    plt.close(figure)


def plot_shot_robustness(robustness: pd.DataFrame, results_dir: Path) -> None:
    order = ["exact", "4096-shot", "1024-shot", "256-shot"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, family in zip(axes, ("1q", "2q")):
        subset = robustness[robustness["family"] == family]
        for model, group in subset.groupby("model"):
            indexed = group.set_index("feature_regime").loc[order]
            axis.plot(
                order,
                indexed["parameter_mae"],
                marker="o",
                label=model,
            )
        axis.set_title(family)
        axis.set_ylabel("parameter MAE")
        axis.tick_params(axis="x", rotation=20)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(results_dir / "shot_robustness.png", dpi=160)
    plt.close(figure)


def plot_qnn_vs_mlp(comparison: pd.DataFrame, results_dir: Path) -> None:
    subset = comparison[comparison["model"].isin(["QNN", "MLP"])]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, family in zip(axes, ("1q", "2q")):
        current = subset[subset["family"] == family]
        axis.bar(current["model"], current["parameter_mae"])
        axis.set_title(f"{family}: no quantum-advantage claim")
        axis.set_ylabel("parameter MAE")
    figure.tight_layout()
    figure.savefig(results_dir / "qnn_vs_mlp.png", dpi=160)
    plt.close(figure)


def plot_compensation(results: list[dict[str, Any]], results_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, result in zip(axes, results):
        rows = result["evaluations"]["QNN"].per_sample
        before = np.asarray([row["fidelity_before_compensation"] for row in rows])
        after = np.asarray([row["fidelity_after_compensation"] for row in rows])
        axis.scatter(before, after, s=20, alpha=0.7)
        lower = min(float(before.min()), float(after.min()))
        axis.plot([lower, 1.0], [lower, 1.0], "k--", linewidth=1)
        axis.set_title(result["family"])
        axis.set_xlabel("fidelity before")
        axis.set_ylabel("fidelity after coherent correction")
    figure.tight_layout()
    figure.savefig(results_dir / "coherent_correction_improvement.png", dpi=160)
    plt.close(figure)


def plot_learning_curves(results: list[dict[str, Any]], results_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, result in zip(axes, results):
        for name in ("QNN", "QNN-shot-robust"):
            history = result["models"][name].history
            axis.plot(
                [row["epoch"] for row in history],
                [row["validation_loss"] for row in history],
                label=name,
            )
        axis.set_title(result["family"])
        axis.set_xlabel("epoch")
        axis.set_ylabel("validation loss")
        axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "learning_curves.png", dpi=160)
    plt.close(figure)


def write_manifest(
    results: list[dict[str, Any]],
    model_dir: Path,
    comparison: pd.DataFrame,
) -> None:
    families = {}
    for result in results:
        family = result["family"]
        families[family] = {
            "architectures": {
                name: model.architecture()
                for name, model in result["models"].items()
                if name in {"QNN", "MLP", "Ridge", "QNN-shot-robust"}
            },
            "test_metrics": {
                name: evaluation.summary
                for name, evaluation in result["evaluations"].items()
            },
            "per_parameter_test_metrics": {
                name: evaluation.per_parameter
                for name, evaluation in result["evaluations"].items()
            },
            "validation_parameter_mae": result["validation_parameter_mae"],
            "best_qnn_validation_loss": {
                "exact": min(
                    row["validation_loss"] for row in result["models"]["QNN"].history
                ),
                "shot_robust_1024": min(
                    row["validation_loss"]
                    for row in result["models"]["QNN-shot-robust"].history
                ),
            },
            "model_paths": {
                key: serialize_manifest_path(Path(path))
                for key, path in result["paths"].items()
            },
            "qiskit_circuit_paths": {
                key: serialize_manifest_path(Path(path))
                for key, path in result["circuits"].items()
            },
        }
    manifest = {
        "schema_version": "1.1",
        "created_by": "scripts/train_models.py",
        "primary_checkpoint_format": (
            "portable NumPy NPZ payload; .pt is retained as the requested "
            "artifact name and does not contain Python pickle"
        ),
        "models": {
            family: {
                "artifact": families[family]["model_paths"]["ridge"],
                "kind": "ridge",
            }
            for family in families
        },
        "families": families,
        "fair_comparison": comparison.to_dict(orient="records"),
        "claims": {
            "quantum_advantage": False,
            "qnn_role": "backend-executed shallow quantum-inference demonstrator",
            "runtime_integration": (
                "diagnostic_features -> trained QNN circuit -> measured Z/ZZ "
                "observables -> saved classical readout"
            ),
        },
        "limitations": [
            (
                "The framework-light QNN uses an exact NumPy statevector, a "
                "seeded stochastic search over circuit angles, and an exact "
                "Ridge readout."
            ),
            (
                "OpenQASM and a Qiskit/Haiqu parameter-binding template are "
                "exported with measurements. Local Aer/fake-backend execution "
                "is integrated; authenticated cloud effects must be reported "
                "from a real Haiqu run."
            ),
            (
                "The 2Q model reconstructs the requested eight gate-specific "
                "coefficients and consumes local kappa values from prior 1Q "
                "calibration."
            ),
            (
                "Exact PTM loss is implemented for evaluation but disabled "
                "during gradient training in the framework-light backend."
            ),
            (
                "Results use synthetic local-generator channels and do not "
                "establish quantum advantage or out-of-distribution robustness."
            ),
        ],
    }
    (model_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    model_dir = PROJECT_ROOT / "artifacts" / "models"
    predictions_dir = PROJECT_ROOT / "artifacts" / "predictions"
    circuit_dir = PROJECT_ROOT / "artifacts" / "circuits"
    results_dir = PROJECT_ROOT / "results" / "models"
    for directory in (
        model_dir,
        predictions_dir,
        circuit_dir,
        results_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    ensure_datasets(
        args.dataset_dir,
        args.profile,
        args.seed,
        allow_generation=not args.skip_data_generation,
    )
    results = [
        train_family(
            family,
            load_family(args.dataset_dir, family),
            args,
            model_dir,
            circuit_dir,
        )
        for family in ("1q", "2q")
    ]
    for result in results:
        write_predictions(result, predictions_dir)

    comparison = pd.DataFrame(comparison_rows(results))
    robustness = pd.DataFrame(
        [row for result in results for row in result["robustness_rows"]]
    )
    comparison.to_csv(results_dir / "model_comparison.csv", index=False)
    robustness.to_csv(results_dir / "shot_robustness.csv", index=False)
    for result in results:
        plot_true_vs_predicted(result, results_dir)
    plot_choi_histograms(results, results_dir)
    plot_shot_robustness(robustness, results_dir)
    plot_qnn_vs_mlp(comparison, results_dir)
    plot_compensation(results, results_dir)
    plot_learning_curves(results, results_dir)
    write_manifest(results, model_dir, comparison)
    print(f"wrote model deliverables to {model_dir}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
