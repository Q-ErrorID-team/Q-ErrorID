"""Dataset and output contracts shared by the model implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ONE_Q_NAMES = (
    "alpha_X",
    "alpha_Y",
    "alpha_Z",
    "gamma_X",
    "gamma_Y",
    "gamma_Z",
    "kappa_down",
)
TWO_Q_NAMES = (
    "alpha_ZI",
    "alpha_IZ",
    "alpha_ZX",
    "alpha_ZZ",
    "gamma_ZI",
    "gamma_IZ",
    "gamma_ZX",
    "gamma_ZZ",
)


@dataclass(frozen=True)
class OutputSpec:
    """Stable ordering and physical bounds for one model family."""

    family: str
    names: tuple[str, ...]
    n_alpha: int
    n_gamma: int
    has_kappa: bool

    @property
    def n_outputs(self) -> int:
        return len(self.names)

    @property
    def lower_bounds(self) -> np.ndarray:
        pieces = [
            np.full(self.n_alpha, -0.15),
            np.zeros(self.n_gamma),
        ]
        if self.has_kappa:
            pieces.append(np.zeros(1))
        return np.concatenate(pieces)

    @property
    def upper_bounds(self) -> np.ndarray:
        pieces = [
            np.full(self.n_alpha, 0.15),
            np.full(self.n_gamma, 0.03),
        ]
        if self.has_kappa:
            pieces.append(np.full(1, 0.05))
        return np.concatenate(pieces)

    def project(self, values: np.ndarray) -> np.ndarray:
        """Project raw regression outputs into the sampled physical domain."""

        array = np.asarray(values, dtype=float)
        return np.clip(array, self.lower_bounds, self.upper_bounds)


def output_spec(family: str) -> OutputSpec:
    normalized = family.lower().replace("_mixed_channel", "")
    if normalized in {"1q", "one_qubit"}:
        return OutputSpec("1q", ONE_Q_NAMES, 3, 3, True)
    if normalized in {"2q", "two_qubit"}:
        return OutputSpec("2q", TWO_Q_NAMES, 4, 4, False)
    raise ValueError(f"Unknown model family: {family!r}")


@dataclass(frozen=True)
class ModelDataset:
    """Validated view of an Agent-1 NPZ dataset."""

    X: np.ndarray
    y: np.ndarray
    known_kappa: np.ndarray
    feature_variants: dict[str, np.ndarray]
    metadata: dict[str, Any]
    spec: OutputSpec

    @property
    def size(self) -> int:
        return int(self.X.shape[0])


def _decode_metadata(value: np.ndarray) -> dict[str, Any]:
    if np.asarray(value).size != 1:
        raise ValueError("Dataset metadata must be a scalar JSON value")
    decoded = json.loads(str(np.asarray(value).item()))
    if not isinstance(decoded, dict):
        raise TypeError("Dataset metadata must decode to an object")
    return decoded


def load_model_dataset(path: str | Path) -> ModelDataset:
    """Load and validate a model-ready view without enabling pickle."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"X_exact", "y_alpha", "y_gamma", "metadata"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{source} is missing required arrays: {sorted(missing)}")
        metadata = _decode_metadata(archive["metadata"])
        features = {
            key: np.asarray(archive[key], dtype=float)
            for key in archive.files
            if key.startswith("X_")
        }
        alpha = np.asarray(archive["y_alpha"], dtype=float)
        gamma = np.asarray(archive["y_gamma"], dtype=float)
        kappa = (
            np.asarray(archive["y_kappa"], dtype=float)
            if "y_kappa" in archive.files
            else np.zeros((alpha.shape[0], 0), dtype=float)
        )

    family_name = str(metadata.get("family", ""))
    if family_name.startswith("1q") or alpha.shape[1] == 3:
        spec = output_spec("1q")
        if kappa.shape != (alpha.shape[0], 1):
            raise ValueError("A 1Q dataset must contain one kappa target")
        targets = np.concatenate([alpha, gamma, kappa], axis=1)
    else:
        spec = output_spec("2q")
        if kappa.shape != (alpha.shape[0], 2):
            raise ValueError("A 2Q dataset must contain two known local damping rates")
        targets = np.concatenate([alpha, gamma], axis=1)

    exact = features["X_exact"]
    if exact.ndim != 2:
        raise ValueError("X_exact must be a two-dimensional feature matrix")
    if targets.shape != (exact.shape[0], spec.n_outputs):
        raise ValueError("Feature and target shapes are inconsistent")
    for name, variant in features.items():
        if variant.shape != exact.shape:
            raise ValueError(
                f"Feature variant {name!r} has shape {variant.shape}, "
                f"expected {exact.shape}"
            )
    return ModelDataset(
        X=exact,
        y=targets,
        known_kappa=kappa,
        feature_variants=features,
        metadata=metadata,
        spec=spec,
    )
