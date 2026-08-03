"""Framework-light hybrid quantum model and physical classical baselines."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import output_spec


@dataclass(frozen=True)
class LossConfig:
    """Weights retained as an explicit training contract."""

    w_alpha: float = 1.0
    w_gamma: float = 1.0
    w_kappa: float = 1.0
    lambda_sparse: float = 0.0
    lambda_classification: float = 0.0


@dataclass(frozen=True)
class QNNTrainingConfig:
    """Small stochastic-search configuration for the quantum feature layer."""

    epochs: int = 45
    batch_size: int = 32
    learning_rate: float = 0.025
    patience: int = 8
    gradient_clip: float = 2.0
    seed: int = 20260724
    shot_robust: bool = False

    def validate(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.patience < 1:
            raise ValueError("patience must be positive")


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.asarray(values, dtype=float)
    return np.where(np.abs(scale) < 1e-10, 1.0, scale)


def _ridge_solution(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    mean = x.mean(axis=0)
    scale = _safe_scale(x.std(axis=0))
    standardized = (x - mean) / scale
    target_mean = y.mean(axis=0)
    centered = y - target_mean
    gram = standardized.T @ standardized
    gram += float(ridge) * np.eye(gram.shape[0])
    coefficient = np.linalg.solve(gram, standardized.T @ centered).T
    return coefficient, target_mean, mean, scale


def randomized_shot_augment(
    features: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add unbiased binomial measurement noise at randomized shot counts."""

    values = np.clip(np.asarray(features, dtype=float), -1.0, 1.0)
    shots = rng.choice(np.array([256, 1024, 4096]), size=values.shape[0])
    probabilities = (values + 1.0) / 2.0
    counts = rng.binomial(shots[:, None], probabilities)
    return 2.0 * counts / shots[:, None] - 1.0


def comparable_hidden_units(
    input_dim: int,
    output_dim: int,
    parameter_budget: int,
) -> int:
    """Choose the largest one-hidden-layer MLP within a parameter budget."""

    per_hidden = int(input_dim) + int(output_dim) + 1
    available = max(0, int(parameter_budget) - int(output_dim))
    return max(1, available // max(per_hidden, 1))


class PhysicalBaseline:
    """Ridge or small MLP with outputs projected to physical bounds."""

    def __init__(
        self,
        family: str,
        kind: str,
        *,
        hidden_units: int = 8,
        seed: int = 0,
        ridge: float = 1e-4,
    ):
        normalized_kind = kind.lower()
        if normalized_kind not in {"ridge", "mlp"}:
            raise ValueError("kind must be 'ridge' or 'mlp'")
        self.family = output_spec(family).family
        self.spec = output_spec(family)
        self.kind = normalized_kind
        self.hidden_units = max(1, int(hidden_units))
        self.seed = int(seed)
        self.ridge = float(ridge)
        self.training_time = 0.0
        self._fitted = False

    def fit(self, features: np.ndarray, targets: np.ndarray) -> PhysicalBaseline:
        start = time.perf_counter()
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or y.shape != (x.shape[0], self.spec.n_outputs):
            raise ValueError("Training feature and target shapes are inconsistent")
        if self.kind == "ridge":
            (
                self.coefficients,
                self.intercept,
                self.feature_mean,
                self.feature_scale,
            ) = _ridge_solution(x, y, ridge=self.ridge)
        else:
            from sklearn.neural_network import MLPRegressor

            self.feature_mean = x.mean(axis=0)
            self.feature_scale = _safe_scale(x.std(axis=0))
            standardized = (x - self.feature_mean) / self.feature_scale
            self._mlp = MLPRegressor(
                hidden_layer_sizes=(self.hidden_units,),
                activation="tanh",
                solver="lbfgs",
                alpha=1e-4,
                max_iter=600,
                random_state=self.seed,
            )
            self._mlp.fit(standardized, y)
        self.training_time = time.perf_counter() - start
        self._fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("The model must be fitted before prediction")
        x = np.asarray(features, dtype=float)
        one_sample = x.ndim == 1
        x = np.atleast_2d(x)
        standardized = (x - self.feature_mean) / self.feature_scale
        if self.kind == "ridge":
            raw = standardized @ self.coefficients.T + self.intercept
        elif hasattr(self, "_loaded_coefs"):
            raw = standardized
            for coefficient, intercept in zip(
                self._loaded_coefs[:-1],
                self._loaded_intercepts[:-1],
            ):
                raw = np.tanh(raw @ coefficient + intercept)
            raw = raw @ self._loaded_coefs[-1] + self._loaded_intercepts[-1]
        else:
            raw = self._mlp.predict(standardized)
        projected = self.spec.project(raw)
        return projected[0] if one_sample else projected

    @property
    def trainable_parameter_count(self) -> int:
        if self.kind == "ridge":
            if not self._fitted:
                return 0
            return int(self.coefficients.size + self.intercept.size)
        if not self._fitted:
            return 0
        if hasattr(self, "_loaded_coefs"):
            return int(
                sum(array.size for array in self._loaded_coefs)
                + sum(array.size for array in self._loaded_intercepts)
            )
        return int(
            sum(array.size for array in self._mlp.coefs_)
            + sum(array.size for array in self._mlp.intercepts_)
        )

    def architecture(self) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "output_names": list(self.spec.names),
            "trainable_parameters": self.trainable_parameter_count,
            "physical_output_projection": {
                name: [float(low), float(high)]
                for name, low, high in zip(
                    self.spec.names,
                    self.spec.lower_bounds,
                    self.spec.upper_bounds,
                )
            },
        }
        if self.kind == "mlp":
            result["hidden_units"] = self.hidden_units
        return result

    def save(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "feature_mean": self.feature_mean,
            "feature_scale": self.feature_scale,
            "metadata": np.array(
                json.dumps(
                    {
                        "family": self.family,
                        "kind": self.kind,
                        "hidden_units": self.hidden_units,
                        "output_names": self.spec.names,
                    },
                    sort_keys=True,
                ),
                dtype=np.str_,
            ),
        }
        if self.kind == "ridge":
            arrays["coef"] = self.coefficients
            arrays["intercept"] = self.intercept
        else:
            arrays["layer_count"] = np.array(len(self._mlp.coefs_))
            for index, value in enumerate(self._mlp.coefs_):
                arrays[f"coef_{index}"] = value
            for index, value in enumerate(self._mlp.intercepts_):
                arrays[f"intercept_{index}"] = value
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> PhysicalBaseline:
        """Restore a portable Ridge or MLP checkpoint without pickle."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
            model = cls(
                metadata["family"],
                metadata["kind"],
                hidden_units=int(metadata.get("hidden_units", 1)),
            )
            model.feature_mean = np.asarray(archive["feature_mean"], dtype=float)
            model.feature_scale = _safe_scale(
                np.asarray(archive["feature_scale"], dtype=float)
            )
            if model.kind == "ridge":
                model.coefficients = np.asarray(archive["coef"], dtype=float)
                model.intercept = np.asarray(archive["intercept"], dtype=float)
            else:
                layer_count = int(np.asarray(archive["layer_count"]).item())
                model._loaded_coefs = tuple(
                    np.asarray(archive[f"coef_{index}"], dtype=float)
                    for index in range(layer_count)
                )
                model._loaded_intercepts = tuple(
                    np.asarray(archive[f"intercept_{index}"], dtype=float)
                    for index in range(layer_count)
                )
                if not model._loaded_coefs:
                    raise ValueError(f"{source} contains no MLP layers")
                model.hidden_units = int(model._loaded_coefs[0].shape[1])
            model._fitted = True
        return model


class HybridQNN:
    """Exact statevector quantum feature layer with a trained physical readout.

    The circuit angles are optimized by a small seeded stochastic search.  For
    every proposed circuit, the classical linear readout is solved exactly.
    This keeps the implementation dependency-light while still training the
    quantum circuit parameters rather than presenting a fixed projection as a
    variational QNN.
    """

    def __init__(
        self,
        input_dim: int,
        family: str,
        *,
        n_qubits: int,
        n_layers: int = 2,
        entanglement: str = "ring",
        auxiliary_classifier: bool = True,
        seed: int = 0,
    ):
        if input_dim < 1 or n_qubits < 1 or n_layers < 1:
            raise ValueError("input_dim, n_qubits, and n_layers must be positive")
        if entanglement != "ring":
            raise ValueError("Only ring entanglement is supported")
        self.input_dim = int(input_dim)
        self.family = output_spec(family).family
        self.spec = output_spec(family)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.entanglement = entanglement
        self.auxiliary_classifier = bool(auxiliary_classifier)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        projection = rng.normal(size=(self.n_layers, self.n_qubits, 2, self.input_dim))
        norms = np.linalg.norm(projection, axis=-1, keepdims=True)
        self.projection = projection / np.where(norms == 0.0, 1.0, norms)
        self.theta = rng.uniform(-0.12, 0.12, size=(self.n_layers, self.n_qubits, 2))
        self.history: list[dict[str, float | int | bool]] = []
        self.training_time = 0.0
        self._fitted = False
        self._measurement_signs = self._build_measurement_signs()
        self._cnot_permutations = self._build_cnot_permutations()

    @property
    def quantum_feature_count(self) -> int:
        return 2 * self.n_qubits

    def _build_measurement_signs(self) -> np.ndarray:
        basis = np.arange(2**self.n_qubits)
        singles = [1.0 - 2.0 * ((basis >> qubit) & 1) for qubit in range(self.n_qubits)]
        correlations = [
            singles[qubit] * singles[(qubit + 1) % self.n_qubits]
            for qubit in range(self.n_qubits)
        ]
        return np.asarray(singles + correlations, dtype=float)

    def _build_cnot_permutations(self) -> tuple[np.ndarray, ...]:
        basis = np.arange(2**self.n_qubits)
        permutations = []
        for control in range(self.n_qubits):
            target = (control + 1) % self.n_qubits
            flip = ((basis >> control) & 1) << target
            permutations.append(basis ^ flip)
        return tuple(permutations)

    @staticmethod
    def _apply_rotations(
        state: np.ndarray,
        *,
        qubit: int,
        ry_angle: np.ndarray,
        rz_angle: np.ndarray,
    ) -> None:
        dimension = state.shape[1]
        mask = 1 << qubit
        zero = np.arange(dimension)
        zero = zero[(zero & mask) == 0]
        one = zero | mask
        amplitude_zero = state[:, zero].copy()
        amplitude_one = state[:, one].copy()
        cosine = np.cos(ry_angle / 2.0)[:, None]
        sine = np.sin(ry_angle / 2.0)[:, None]
        phase_zero = np.exp(-0.5j * rz_angle)[:, None]
        phase_one = np.exp(0.5j * rz_angle)[:, None]
        state[:, zero] = (cosine * amplitude_zero - sine * amplitude_one) * phase_zero
        state[:, one] = (sine * amplitude_zero + cosine * amplitude_one) * phase_one

    def _encode_angles(self, standardized: np.ndarray) -> np.ndarray:
        raw = np.einsum(
            "bf,lqrf->blqr",
            standardized,
            self.projection,
            optimize=True,
        )
        return np.pi * np.tanh(raw)

    def _quantum_features(
        self,
        standardized: np.ndarray,
        theta: np.ndarray | None = None,
    ) -> np.ndarray:
        values = np.asarray(standardized, dtype=float)
        angles = self._encode_angles(values)
        parameters = self.theta if theta is None else np.asarray(theta, dtype=float)
        state = np.zeros((values.shape[0], 2**self.n_qubits), dtype=np.complex128)
        state[:, 0] = 1.0
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                self._apply_rotations(
                    state,
                    qubit=qubit,
                    ry_angle=angles[:, layer, qubit, 0] + parameters[layer, qubit, 0],
                    rz_angle=angles[:, layer, qubit, 1] + parameters[layer, qubit, 1],
                )
            for permutation in self._cnot_permutations:
                state = state[:, permutation]
        probabilities = np.abs(state) ** 2
        return probabilities @ self._measurement_signs.T

    @staticmethod
    def _head_predict(
        features: np.ndarray,
        coefficient: np.ndarray,
        intercept: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        standardized = (features - mean) / scale
        return standardized @ coefficient.T + intercept

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        *,
        config: QNNTrainingConfig,
        loss_config: LossConfig | None = None,
    ) -> HybridQNN:
        del loss_config  # The physical projection already enforces rate domains.
        config.validate()
        start = time.perf_counter()
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        x_val = np.asarray(validation_features, dtype=float)
        y_val = np.asarray(validation_targets, dtype=float)
        if x.shape[1] != self.input_dim or x_val.shape[1] != self.input_dim:
            raise ValueError("Feature dimension does not match the QNN")
        if y.shape[1] != self.spec.n_outputs:
            raise ValueError("Target dimension does not match the QNN family")
        self.feature_mean = x.mean(axis=0)
        self.feature_scale = _safe_scale(x.std(axis=0))
        train_standardized = (x - self.feature_mean) / self.feature_scale
        val_standardized = (x_val - self.feature_mean) / self.feature_scale
        rng = np.random.default_rng(config.seed)

        best_theta = self.theta.copy()
        best_state: tuple[np.ndarray, ...] | None = None
        best_loss = float("inf")
        stale_epochs = 0
        self.history = []
        for epoch in range(1, config.epochs + 1):
            if epoch == 1:
                candidate = best_theta
            else:
                step = config.learning_rate / np.sqrt(epoch)
                perturbation = np.clip(
                    rng.normal(scale=step, size=best_theta.shape),
                    -config.gradient_clip * step,
                    config.gradient_clip * step,
                )
                candidate = best_theta + perturbation
            train_quantum = self._quantum_features(train_standardized, candidate)
            coefficient, intercept, q_mean, q_scale = _ridge_solution(
                train_quantum,
                y,
                ridge=2e-4,
            )
            val_quantum = self._quantum_features(val_standardized, candidate)
            raw = self._head_predict(
                val_quantum,
                coefficient,
                intercept,
                q_mean,
                q_scale,
            )
            prediction = self.spec.project(raw)
            validation_loss = float(np.mean((prediction - y_val) ** 2))
            improved = validation_loss + 1e-12 < best_loss
            if improved:
                best_loss = validation_loss
                best_theta = candidate.copy()
                best_state = (
                    coefficient,
                    intercept,
                    q_mean,
                    q_scale,
                )
                stale_epochs = 0
            else:
                stale_epochs += 1
            self.history.append(
                {
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "candidate_validation_loss": validation_loss,
                    "accepted": improved,
                }
            )
            if stale_epochs >= config.patience:
                break

        if best_state is None:
            raise RuntimeError("QNN training did not produce a valid state")
        self.theta = best_theta
        (
            self.coefficients,
            self.intercept,
            self.quantum_feature_mean,
            self.quantum_feature_scale,
        ) = best_state
        self._fitted = True
        self.training_time = time.perf_counter() - start
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("The model must be fitted before prediction")
        x = np.asarray(features, dtype=float)
        one_sample = x.ndim == 1
        x = np.atleast_2d(x)
        standardized = (x - self.feature_mean) / self.feature_scale
        quantum = self._quantum_features(standardized)
        raw = self._head_predict(
            quantum,
            self.coefficients,
            self.intercept,
            self.quantum_feature_mean,
            self.quantum_feature_scale,
        )
        projected = self.spec.project(raw)
        return projected[0] if one_sample else projected

    def data_angles(self, features: np.ndarray) -> np.ndarray:
        """Bind physical diagnostic features to the trained circuit angles."""

        if not self._fitted:
            raise RuntimeError("The model must be fitted before angle binding")
        values = np.asarray(features, dtype=float)
        one_sample = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != self.input_dim:
            raise ValueError(
                f"QNN expects {self.input_dim} features, got {values.shape[1]}"
            )
        standardized = (values - self.feature_mean) / self.feature_scale
        angles = self._encode_angles(standardized)
        return angles[0] if one_sample else angles

    def build_qiskit_inference_circuit(
        self,
        features: np.ndarray,
        *,
        measure: bool = True,
        name: str | None = None,
    ):
        """Build the trained, feature-bound QNN circuit used for deployment."""

        from qiskit import QuantumCircuit

        angles = np.asarray(self.data_angles(features), dtype=float)
        if angles.shape != (self.n_layers, self.n_qubits, 2):
            raise ValueError("Exactly one feature sample is required per QNN circuit")
        circuit = QuantumCircuit(
            self.n_qubits,
            self.n_qubits if measure else 0,
            name=name or f"q_error_id_{self.family}_qnn",
        )
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                circuit.ry(
                    float(angles[layer, qubit, 0] + self.theta[layer, qubit, 0]),
                    qubit,
                )
                circuit.rz(
                    float(angles[layer, qubit, 1] + self.theta[layer, qubit, 1]),
                    qubit,
                )
            for qubit in range(self.n_qubits):
                circuit.cx(qubit, (qubit + 1) % self.n_qubits)
        if measure:
            circuit.measure(range(self.n_qubits), range(self.n_qubits))
        circuit.metadata = {
            "model_kind": "qnn",
            "model_family": self.family,
            "trained_parameters": True,
            "readout_observables": self.architecture()["measurements"],
        }
        return circuit

    def quantum_features_from_distribution(
        self,
        distribution: Mapping[str, float | int],
    ) -> np.ndarray:
        """Extract the trained Z and nearest-neighbour ZZ readout features."""

        total = float(sum(float(value) for value in distribution.values()))
        if total <= 0.0:
            raise ValueError("QNN measurement distribution is empty")
        singles = np.zeros(self.n_qubits, dtype=float)
        correlations = np.zeros(self.n_qubits, dtype=float)
        for raw_bits, raw_weight in distribution.items():
            bits = str(raw_bits).replace(" ", "")
            if len(bits) < self.n_qubits:
                bits = bits.zfill(self.n_qubits)
            weight = float(raw_weight) / total
            signs = np.asarray(
                [
                    1.0 - 2.0 * int(bits[-1 - qubit])
                    for qubit in range(self.n_qubits)
                ],
                dtype=float,
            )
            singles += weight * signs
            correlations += weight * signs * np.roll(signs, -1)
        return np.concatenate([singles, correlations])

    def predict_from_quantum_features(
        self,
        quantum_features: np.ndarray,
    ) -> np.ndarray:
        """Apply the saved classical readout to measured QNN observables."""

        if not self._fitted:
            raise RuntimeError("The model must be fitted before prediction")
        values = np.asarray(quantum_features, dtype=float)
        one_sample = values.ndim == 1
        values = np.atleast_2d(values)
        if values.shape[1] != self.quantum_feature_count:
            raise ValueError(
                "QNN readout expects "
                f"{self.quantum_feature_count} observables, got {values.shape[1]}"
            )
        raw = self._head_predict(
            values,
            self.coefficients,
            self.intercept,
            self.quantum_feature_mean,
            self.quantum_feature_scale,
        )
        projected = self.spec.project(raw)
        return projected[0] if one_sample else projected

    @property
    def trainable_parameter_count(self) -> int:
        circuit = self.theta.size
        if not self._fitted:
            return int(circuit)
        return int(circuit + self.coefficients.size + self.intercept.size)

    def architecture(self) -> dict[str, Any]:
        return {
            "backend": "numpy_exact_statevector",
            "training": "stochastic_variational_search_with_exact_ridge_readout",
            "input_dim": self.input_dim,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "encoding": "angle_data_reuploading",
            "entanglement": self.entanglement,
            "measurements": [
                *[f"Z{q}" for q in range(self.n_qubits)],
                *[f"Z{q}Z{(q + 1) % self.n_qubits}" for q in range(self.n_qubits)],
            ],
            "output_names": list(self.spec.names),
            "circuit_parameters": int(self.theta.size),
            "trainable_parameters": self.trainable_parameter_count,
            "auxiliary_classifier": self.auxiliary_classifier,
        }

    def save(self, path: str | Path) -> Path:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "1.1",
            "family": self.family,
            "input_dim": self.input_dim,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "entanglement": self.entanglement,
            "auxiliary_classifier": self.auxiliary_classifier,
            "output_names": self.spec.names,
        }
        with destination.open("wb") as stream:
            np.savez_compressed(
                stream,
                projection=self.projection,
                theta=self.theta,
                coef=self.coefficients,
                intercept=self.intercept,
                feature_mean=self.feature_mean,
                feature_scale=self.feature_scale,
                quantum_feature_mean=self.quantum_feature_mean,
                quantum_feature_scale=self.quantum_feature_scale,
                metadata=np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_),
            )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> HybridQNN:
        """Restore the trained quantum circuit and its classical readout."""

        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            required = {
                "projection",
                "theta",
                "coef",
                "intercept",
                "feature_mean",
                "feature_scale",
                "quantum_feature_mean",
                "quantum_feature_scale",
                "metadata",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(
                    f"{source} is missing QNN checkpoint arrays: {sorted(missing)}"
                )
            metadata = json.loads(str(np.asarray(archive["metadata"]).item()))
            model = cls(
                int(metadata["input_dim"]),
                metadata["family"],
                n_qubits=int(metadata["n_qubits"]),
                n_layers=int(metadata["n_layers"]),
                entanglement=str(metadata.get("entanglement", "ring")),
                auxiliary_classifier=bool(
                    metadata.get("auxiliary_classifier", True)
                ),
            )
            model.projection = np.asarray(archive["projection"], dtype=float)
            model.theta = np.asarray(archive["theta"], dtype=float)
            model.coefficients = np.asarray(archive["coef"], dtype=float)
            model.intercept = np.asarray(archive["intercept"], dtype=float)
            model.feature_mean = np.asarray(archive["feature_mean"], dtype=float)
            model.feature_scale = _safe_scale(
                np.asarray(archive["feature_scale"], dtype=float)
            )
            model.quantum_feature_mean = np.asarray(
                archive["quantum_feature_mean"],
                dtype=float,
            )
            model.quantum_feature_scale = _safe_scale(
                np.asarray(archive["quantum_feature_scale"], dtype=float)
            )
            expected_projection = (
                model.n_layers,
                model.n_qubits,
                2,
                model.input_dim,
            )
            if model.projection.shape != expected_projection:
                raise ValueError(
                    f"{source} has projection shape {model.projection.shape}, "
                    f"expected {expected_projection}"
                )
            if model.theta.shape != expected_projection[:3]:
                raise ValueError(
                    f"{source} has theta shape {model.theta.shape}, "
                    f"expected {expected_projection[:3]}"
                )
            model._fitted = True
        return model
