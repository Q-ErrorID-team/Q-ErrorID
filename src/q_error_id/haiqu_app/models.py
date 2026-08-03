"""Agent 2 Ridge/QNN discovery with transparent fallback semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from q_error_id.models.estimators import HybridQNN

ONE_Q_LABELS = ("X", "Y", "Z")
TWO_Q_LABELS = ("ZI", "IZ", "ZX", "ZZ")


@dataclass
class ChannelEstimate:
    """Interpretable local generator returned by Agent 2 or a fallback model."""

    alpha: dict[str, float]
    gamma: dict[str, float]
    kappa_down: dict[str, float]
    model_source: str
    trained_model: bool

    @property
    def coherent_magnitude(self) -> float:
        return float(np.linalg.norm(list(self.alpha.values())))

    @property
    def stochastic_magnitude(self) -> float:
        return float(np.linalg.norm(list(self.gamma.values())))

    @property
    def amplitude_damping(self) -> float:
        return float(max(self.kappa_down.values(), default=0.0))

    @property
    def dominant_pauli_generator(self) -> str:
        combined = {f"alpha_{key}": abs(value) for key, value in self.alpha.items()}
        combined.update(
            {f"gamma_{key}": abs(value) for key, value in self.gamma.items()}
        )
        return max(combined, key=combined.get) if combined else "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "kappa_down": self.kappa_down,
            "coherent_magnitude": self.coherent_magnitude,
            "stochastic_magnitude": self.stochastic_magnitude,
            "amplitude_damping": self.amplitude_damping,
            "dominant_pauli_generator": self.dominant_pauli_generator,
            "model_source": self.model_source,
            "trained_model": self.trained_model,
        }


@dataclass
class NumpyLinearModel:
    coefficients: np.ndarray
    intercept: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    source: str

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float).reshape(1, -1)
        if values.shape[1] != self.feature_mean.size:
            raise ValueError(
                f"Model expects {self.feature_mean.size} features, got {values.shape[1]}"
            )
        standardized = (values - self.feature_mean) / self.feature_scale
        return (standardized @ self.coefficients.T + self.intercept).reshape(-1)

    @classmethod
    def load(cls, path: Path) -> NumpyLinearModel:
        with np.load(path, allow_pickle=False) as archive:
            required = {"coef", "intercept"}
            if not required.issubset(archive.files):
                raise ValueError(f"{path} is not a supported linear model artifact")
            coefficients = np.asarray(archive["coef"], dtype=float)
            intercept = np.asarray(archive["intercept"], dtype=float).reshape(-1)
            feature_mean = (
                np.asarray(archive["feature_mean"], dtype=float).reshape(-1)
                if "feature_mean" in archive.files
                else np.zeros(coefficients.shape[1])
            )
            feature_scale = (
                np.asarray(archive["feature_scale"], dtype=float).reshape(-1)
                if "feature_scale" in archive.files
                else np.ones(coefficients.shape[1])
            )
        return cls(
            coefficients=coefficients,
            intercept=intercept,
            feature_mean=feature_mean,
            feature_scale=np.where(feature_scale == 0.0, 1.0, feature_scale),
            source=f"artifact:{path.name}",
        )

    def save(self, path: Path) -> Path:
        """Save a portable non-pickle deployment artifact."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(
                stream,
                coef=self.coefficients,
                intercept=self.intercept,
                feature_mean=self.feature_mean,
                feature_scale=self.feature_scale,
                source=np.asarray(self.source),
            )
        return path

    @classmethod
    def fit_dataset(
        cls,
        path: Path,
        *,
        family: str,
        ridge: float = 1e-4,
        shots: int = 4096,
        feature_key: str | None = None,
        readout_corrected: bool = False,
    ) -> NumpyLinearModel:
        with np.load(path, allow_pickle=False) as archive:
            requested = f"X_shot_{shots}"
            shot_keys = [
                key
                for key in archive.files
                if key.startswith("X_shot_") and key.removeprefix("X_shot_").isdigit()
            ]
            selected_feature_key = feature_key or requested
            if selected_feature_key not in archive.files and feature_key is not None:
                raise ValueError(f"{path} does not contain {feature_key}")
            if selected_feature_key not in archive.files and shot_keys:
                selected_feature_key = min(
                    shot_keys,
                    key=lambda key: abs(
                        np.log2(int(key.removeprefix("X_shot_")) / max(int(shots), 1))
                    ),
                )
            if selected_feature_key not in archive.files:
                selected_feature_key = "X_exact"
            x = np.asarray(archive[selected_feature_key], dtype=float)
            source_feature = selected_feature_key
            if readout_corrected:
                required = {"X_exact", "X_readout_exact"}
                if not required.issubset(archive.files):
                    raise ValueError(
                        f"{path} lacks exact/readout feature pairs for calibration"
                    )
                exact = np.asarray(archive["X_exact"], dtype=float)
                readout_exact = np.asarray(
                    archive["X_readout_exact"],
                    dtype=float,
                )
                metadata = json.loads(str(archive["metadata"].item()))
                confusion = metadata.get("readout_confusion", {})
                p_plus_to_minus = float(
                    confusion.get("p_plus_to_minus", 0.0)
                )
                p_minus_to_plus = float(
                    confusion.get("p_minus_to_plus", 0.0)
                )
                scale = 1.0 - p_plus_to_minus - p_minus_to_plus
                if family == "1q":
                    from q_error_id.core.protocols import one_qubit_protocol

                    protocol = one_qubit_protocol()
                else:
                    from q_error_id.core.protocols import two_qubit_protocol

                    protocol = two_qubit_protocol(
                        gate_name="CX",
                        basis=TWO_Q_LABELS,
                        target_features=80,
                    )
                observable_scale = np.asarray(
                    [
                        scale
                        ** sum(
                            letter != "I"
                            for letter in protocol.observable_labels[
                                observable_index
                            ]
                        )
                        for _, observable_index in protocol.settings
                    ],
                    dtype=float,
                )
                x = exact + (x - readout_exact) / observable_scale
                x = np.clip(x, -1.0, 1.0)
                source_feature = (
                    f"calibrated({selected_feature_key},X_readout_exact)"
                )
            alpha = np.asarray(archive["y_alpha"], dtype=float)
            gamma = np.asarray(archive["y_gamma"], dtype=float)
            targets = [alpha, gamma]
            if family == "1q" and "y_kappa" in archive.files:
                targets.append(np.asarray(archive["y_kappa"], dtype=float))
            y = np.concatenate(targets, axis=1)
        feature_mean = x.mean(axis=0)
        feature_scale = x.std(axis=0)
        feature_scale[feature_scale < 1e-10] = 1.0
        z = (x - feature_mean) / feature_scale
        target_mean = y.mean(axis=0)
        centered_y = y - target_mean
        gram = z.T @ z + ridge * np.eye(z.shape[1])
        coefficients = np.linalg.solve(gram, z.T @ centered_y).T
        return cls(
            coefficients=coefficients,
            intercept=target_mean,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            source=f"on_demand_ridge:{path.name}:{source_feature}",
        )


class ModelRepository:
    """Load trained Ridge and QNN artifacts with explicit fallback semantics."""

    def __init__(self, model_root: Path, data_root: Path, shots: int = 4096):
        self.model_root = Path(model_root)
        self.data_root = Path(data_root)
        self.shots = int(shots)
        self._cache: dict[
            tuple[str, str],
            NumpyLinearModel | HybridQNN | None,
        ] = {}

    def _manifest_candidate(self, family: str, model_kind: str) -> Path | None:
        for name in ("model_manifest.json", "manifest.json"):
            path = self.model_root / name
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            models = payload.get("models", payload)
            if model_kind == "ridge" and isinstance(models, dict):
                item = models.get(family) or models.get(
                    "1q_mixed_channel" if family == "1q" else "2q_mixed_channel"
                )
                if isinstance(item, str):
                    return self.model_root / item
                if isinstance(item, dict):
                    artifact = item.get("artifact") or item.get("path")
                    if artifact:
                        return self._resolve_artifact(str(artifact))
            families = payload.get("families")
            if isinstance(families, dict):
                item = families.get(family)
                if isinstance(item, dict):
                    paths = item.get("model_paths", {})
                    if not isinstance(paths, dict):
                        continue
                    preferred = (
                        ("ridge",)
                        if model_kind == "ridge"
                        else ("qnn_primary", "qnn_exact")
                    )
                    for key in preferred:
                        if paths.get(key):
                            return self._resolve_artifact(str(paths[key]))
        return None

    def _resolve_artifact(self, artifact: str) -> Path:
        candidate = Path(artifact)
        if candidate.is_absolute():
            return candidate
        direct = self.model_root / candidate
        if direct.exists():
            return direct
        # Manifests commonly store paths relative to the project root, while
        # model_root itself is <project>/artifacts/models.
        project_relative = self.model_root.parents[1] / candidate
        return project_relative if project_relative.exists() else direct

    def _fit_dataset(
        self,
        family: str,
        *,
        feature_regime: str = "raw_shot",
    ) -> NumpyLinearModel | None:
        dataset = self.data_root / (
            "1q_mixed_channel_train.npz"
            if family == "1q"
            else "2q_mixed_channel_train.npz"
        )
        if not dataset.exists():
            return None
        return NumpyLinearModel.fit_dataset(
            dataset,
            family=family,
            shots=self.shots,
            feature_key=None,
            readout_corrected=feature_regime == "readout_corrected",
        )

    def _load_ridge(
        self,
        family: str,
        *,
        feature_regime: str = "raw_shot",
    ) -> NumpyLinearModel | None:
        if feature_regime not in {"raw_shot", "readout_corrected"}:
            raise ValueError(
                "feature_regime must be 'raw_shot' or 'readout_corrected'"
            )
        cache_key = ("ridge", f"{family}:{feature_regime}")
        cached = self._cache.get(cache_key)
        if isinstance(cached, NumpyLinearModel):
            return cached
        if cache_key in self._cache:
            return None
        available_shots = (256, 1024, 4096, 8192)
        closest_shots = min(
            available_shots,
            key=lambda value: abs(np.log2(value / max(self.shots, 1))),
        )
        candidates: list[Path] = []
        if feature_regime == "readout_corrected":
            candidates.append(
                self.model_root
                / f"ridge_{family}_{self.shots}_readout_corrected.npz"
            )
        else:
            candidates.append(
                self.model_root / f"ridge_{family}_{self.shots}.npz"
            )
            if closest_shots != self.shots:
                candidates.append(
                    self.model_root / f"ridge_{family}_{closest_shots}.npz"
                )
            if candidate := self._manifest_candidate(family, "ridge"):
                candidates.append(candidate)
        candidates.extend(
            [
                self.model_root / f"{family}_ridge.npz",
                self.model_root / f"{family}_model.npz",
                self.model_root / f"{family}_linear.npz",
            ]
        )
        for path in candidates:
            if path.exists():
                model = NumpyLinearModel.load(path)
                self._cache[cache_key] = model
                return model

        model = self._fit_dataset(family, feature_regime=feature_regime)
        if model is not None:
            self._cache[cache_key] = model
            return model
        self._cache[cache_key] = None
        return None

    def _load_qnn(self, family: str) -> HybridQNN | None:
        cache_key = ("qnn", family)
        cached = self._cache.get(cache_key)
        if isinstance(cached, HybridQNN):
            return cached
        if cache_key in self._cache:
            return None
        candidates = []
        if candidate := self._manifest_candidate(family, "qnn"):
            candidates.append(candidate)
        candidates.extend(
            [
                self.model_root / f"qnn_{family}.pt",
                self.model_root / f"qnn_{family}_exact.pt",
            ]
        )
        for path in candidates:
            if path.exists():
                model = HybridQNN.load(path)
                model.artifact_source = path
                self._cache[cache_key] = model
                return model
        self._cache[cache_key] = None
        return None

    def qnn_model(self, family: str, *, required: bool = False) -> HybridQNN | None:
        """Return the restored trained QNN, optionally failing if unavailable."""

        model = self._load_qnn(family)
        if model is None and required:
            raise FileNotFoundError(
                f"No trained {family} QNN artifact exists under {self.model_root}"
            )
        return model

    def qnn_circuit(
        self,
        family: str,
        features: np.ndarray,
        *,
        name: str,
        physical_qubits: tuple[int, ...] | None = None,
    ):
        """Build a measured circuit from the restored trained QNN artifact."""

        model = self.qnn_model(family, required=True)
        circuit = model.build_qiskit_inference_circuit(
            features,
            measure=True,
            name=name,
        )
        circuit.metadata = {
            **(circuit.metadata or {}),
            "physical_qubits": list(physical_qubits or range(model.n_qubits)),
            "artifact": Path(model.artifact_source).name,
        }
        return circuit

    def predict_qnn_distribution(
        self,
        family: str,
        distribution: dict[str, float | int],
        *,
        execution_source: str,
    ) -> tuple[ChannelEstimate, np.ndarray]:
        """Decode measured trained-QNN observables and apply its saved readout."""

        model = self.qnn_model(family, required=True)
        quantum_features = model.quantum_features_from_distribution(distribution)
        prediction = model.predict_from_quantum_features(quantum_features)
        source = (
            f"artifact:{Path(model.artifact_source).name}:"
            f"measured_qnn:{execution_source}"
        )
        return (
            self._decode_prediction(family, prediction, source),
            quantum_features,
        )

    @staticmethod
    def _decode_prediction(
        family: str, prediction: np.ndarray, source: str
    ) -> ChannelEstimate:
        values = np.asarray(prediction, dtype=float).reshape(-1)
        labels = ONE_Q_LABELS if family == "1q" else TWO_Q_LABELS
        n = len(labels)
        if values.size < 2 * n:
            raise ValueError(f"{family} model returned only {values.size} values")
        alpha = dict(zip(labels, np.clip(values[:n], -0.15, 0.15)))
        gamma = dict(zip(labels, np.clip(values[n : 2 * n], 0.0, 0.03)))
        kappa: dict[str, float] = {}
        if family == "1q":
            value = float(values[2 * n]) if values.size > 2 * n else 0.0
            kappa = {"down": float(np.clip(value, 0.0, 0.05))}
        return ChannelEstimate(
            alpha={key: float(value) for key, value in alpha.items()},
            gamma={key: float(value) for key, value in gamma.items()},
            kappa_down=kappa,
            model_source=source,
            trained_model=True,
        )

    def predict(
        self,
        family: str,
        features: np.ndarray,
        *,
        model_kind: str = "ridge",
        ideal_features: np.ndarray | None = None,
        feature_regime: str = "raw_shot",
    ) -> ChannelEstimate:
        normalized_kind = model_kind.lower()
        if normalized_kind not in {"ridge", "qnn"}:
            raise ValueError("model_kind must be 'ridge' or 'qnn'")
        if normalized_kind == "qnn":
            qnn = self._load_qnn(family)
            if qnn is not None:
                return self._decode_prediction(
                    family,
                    qnn.predict(features),
                    f"artifact:{Path(qnn.artifact_source).name}:numpy_exact_statevector",
                )
            if family == "1q":
                return analytic_one_qubit_fallback(features)
            return projection_two_qubit_fallback(features, ideal_features)

        model = self._load_ridge(family, feature_regime=feature_regime)
        if model is not None:
            try:
                return self._decode_prediction(
                    family,
                    model.predict(features),
                    model.source,
                )
            except ValueError:
                # A stale export must not suppress a compatible Agent-1-backed
                # baseline. Refit once and replace the incompatible cache item.
                compatible = self._fit_dataset(
                    family,
                    feature_regime=feature_regime,
                )
                if compatible is not None:
                    self._cache[("ridge", f"{family}:{feature_regime}")] = compatible
                    return self._decode_prediction(
                        family,
                        compatible.predict(features),
                        compatible.source,
                    )
        if family == "1q":
            return analytic_one_qubit_fallback(features)
        return projection_two_qubit_fallback(features, ideal_features)


def analytic_one_qubit_fallback(features: np.ndarray) -> ChannelEstimate:
    """Near-identity Bloch-affine reconstruction for the fixed 18 features."""

    values = np.asarray(features, dtype=float).reshape(-1)
    if values.size != 18:
        raise ValueError("The analytic 1Q fallback requires 18 features")
    outputs = values.reshape(6, 3)
    transfer = np.column_stack(
        [
            0.5 * (outputs[0] - outputs[1]),
            0.5 * (outputs[2] - outputs[3]),
            0.5 * (outputs[4] - outputs[5]),
        ]
    )
    offset = np.mean(
        [
            0.5 * (outputs[0] + outputs[1]),
            0.5 * (outputs[2] + outputs[3]),
            0.5 * (outputs[4] + outputs[5]),
        ],
        axis=0,
    )
    generator = transfer - np.eye(3)
    alpha = {
        "X": 0.5 * (generator[2, 1] - generator[1, 2]),
        "Y": 0.5 * (generator[0, 2] - generator[2, 0]),
        "Z": 0.5 * (generator[1, 0] - generator[0, 1]),
    }
    kappa = float(np.clip(offset[2], 0.0, 0.05))
    decay = np.clip(1.0 - np.diag(transfer), 0.0, 1.0)
    sx = max(0.0, float(decay[0] - 0.5 * kappa))
    sy = max(0.0, float(decay[1] - 0.5 * kappa))
    sz = max(0.0, float(decay[2] - kappa))
    gamma = {
        "X": max(0.0, (sy + sz - sx) / 4.0),
        "Y": max(0.0, (sx + sz - sy) / 4.0),
        "Z": max(0.0, (sx + sy - sz) / 4.0),
    }
    return ChannelEstimate(
        alpha={key: float(np.clip(value, -0.15, 0.15)) for key, value in alpha.items()},
        gamma={key: float(np.clip(value, 0.0, 0.03)) for key, value in gamma.items()},
        kappa_down={"down": kappa},
        model_source="analytic_bloch_near_identity_fallback",
        trained_model=False,
    )


def projection_two_qubit_fallback(
    features: np.ndarray,
    ideal_features: np.ndarray | None,
) -> ChannelEstimate:
    """Marked demo-only projection used when neither Agent 2 nor data exist."""

    values = np.asarray(features, dtype=float).reshape(-1)
    reference = (
        np.asarray(ideal_features, dtype=float).reshape(-1)
        if ideal_features is not None
        else np.zeros_like(values)
    )
    if reference.size != values.size:
        raise ValueError("ideal_features have the wrong size")
    delta = values - reference
    indices = np.arange(delta.size)
    alpha = {}
    gamma = {}
    for axis, label in enumerate(TWO_Q_LABELS):
        phase = np.sin((axis + 1) * (indices + 1) * np.pi / 7.0)
        alpha[label] = float(
            np.clip(np.dot(delta, phase) / max(delta.size, 1), -0.15, 0.15)
        )
        subset = delta[indices % len(TWO_Q_LABELS) == axis]
        gamma[label] = float(
            np.clip(
                0.25 * np.mean(np.abs(subset)) if subset.size else 0.0,
                0.0,
                0.03,
            )
        )
    return ChannelEstimate(
        alpha=alpha,
        gamma=gamma,
        kappa_down={},
        model_source="untrained_diagnostic_projection_fallback",
        trained_model=False,
    )


def predict_features_with_agent1(
    estimate: ChannelEstimate,
    family: str,
) -> np.ndarray | None:
    """Forward-predict diagnostic features through Agent 1 when it is available."""

    try:
        from q_error_id.core.channels import build_channel
        from q_error_id.core.parameters import (
            one_qubit_parameters,
            two_qubit_parameters,
        )
        from q_error_id.core.protocols import (
            extract_features,
            one_qubit_protocol,
            two_qubit_protocol,
        )
    except ImportError:
        return None

    if family == "1q":
        parameters = one_qubit_parameters(
            alpha=np.asarray([estimate.alpha[x] for x in ONE_Q_LABELS]),
            gamma=np.asarray([estimate.gamma[x] for x in ONE_Q_LABELS]),
            kappa_down=estimate.kappa_down.get("down", 0.0),
        )
        protocol = one_qubit_protocol()
    else:
        parameters = two_qubit_parameters(
            gate_name="CX",
            basis=TWO_Q_LABELS,
            alpha=np.asarray([estimate.alpha[x] for x in TWO_Q_LABELS]),
            gamma=np.asarray([estimate.gamma[x] for x in TWO_Q_LABELS]),
            kappa_down=np.zeros(2),
        )
        protocol = two_qubit_protocol(
            gate_name="CX",
            basis=TWO_Q_LABELS,
            target_features=80,
        )
    if family == "2q":
        from q_error_id.core.channels import build_diagnostic_channel

        channel = build_diagnostic_channel(parameters)
    else:
        channel = build_channel(parameters)
    return np.asarray(extract_features(channel, protocol))
