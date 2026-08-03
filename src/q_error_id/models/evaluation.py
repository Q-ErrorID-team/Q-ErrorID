"""Channel-aware evaluation of learned local generator parameters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from q_error_id.core import (
    ChannelParameters,
    build_channel,
    build_diagnostic_channel,
    choi_state_fidelity,
    identity_channel,
    one_qubit_parameters,
    relative_ptm_frobenius_error,
    two_qubit_parameters,
)
from q_error_id.core.protocols import (
    extract_features,
    one_qubit_protocol,
    two_qubit_protocol,
)

from .contracts import OutputSpec


@dataclass(frozen=True)
class EvaluationResult:
    predictions: np.ndarray
    summary: dict[str, float]
    per_parameter: dict[str, dict[str, float]]
    per_sample: list[dict[str, Any]]


def vector_to_parameters(
    values: np.ndarray,
    spec: OutputSpec,
    *,
    known_kappa: np.ndarray | None = None,
) -> ChannelParameters:
    """Convert a model output vector into the shared physical contract."""

    vector = spec.project(np.asarray(values, dtype=float).reshape(-1))
    if vector.size != spec.n_outputs:
        raise ValueError(
            f"{spec.family} expects {spec.n_outputs} outputs, got {vector.size}"
        )
    if spec.family == "1q":
        return one_qubit_parameters(
            alpha=vector[: spec.n_alpha],
            gamma=vector[spec.n_alpha : spec.n_alpha + spec.n_gamma],
            kappa_down=vector[-1],
        )
    kappa = (
        np.zeros(2, dtype=float)
        if known_kappa is None
        else np.asarray(known_kappa, dtype=float).reshape(-1)
    )
    return two_qubit_parameters(
        gate_name="CX",
        basis=("ZI", "IZ", "ZX", "ZZ"),
        alpha=vector[: spec.n_alpha],
        gamma=vector[spec.n_alpha : spec.n_alpha + spec.n_gamma],
        kappa_down=kappa,
    )


def recommend_strategy(parameters: ChannelParameters) -> str:
    """Return an interpretable correction/mitigation recommendation."""

    coherent = float(np.linalg.norm(parameters.alpha))
    stochastic = float(np.sum(parameters.gamma))
    damping = float(
        np.sum(parameters.kappa_down)
        if parameters.kappa_down is not None
        else 0.0
    )
    if max(coherent, stochastic, damping) <= 1e-10:
        return "no_update"
    if coherent >= max(stochastic, damping):
        return "coherent_inverse_then_mitigation"
    if damping >= stochastic:
        return "relaxation_aware_remapping_or_rescheduling"
    return "stochastic_error_mitigation"


def _r2_score(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = float(np.sum((target - prediction) ** 2))
    centered = float(np.sum((target - np.mean(target)) ** 2))
    return 1.0 - residual / centered if centered > 1e-15 else 0.0


def _parameters(
    spec: OutputSpec,
    values: np.ndarray,
    known_kappa: np.ndarray,
):
    if spec.family == "1q":
        return one_qubit_parameters(
            alpha=values[:3],
            gamma=values[3:6],
            kappa_down=values[6],
        )
    return two_qubit_parameters(
        gate_name="CX",
        basis=("ZI", "IZ", "ZX", "ZZ"),
        alpha=values[:4],
        gamma=values[4:8],
        kappa_down=known_kappa,
    )


def _dominant_family(
    spec: OutputSpec,
    values: np.ndarray,
    known_kappa: np.ndarray,
) -> str:
    strengths = {
        "coherent": float(np.linalg.norm(values[: spec.n_alpha])),
        "stochastic": float(np.sum(values[spec.n_alpha : spec.n_alpha + spec.n_gamma])),
        "relaxation": float(values[-1] if spec.has_kappa else np.sum(known_kappa)),
    }
    return max(strengths, key=strengths.get)


def evaluate_model(
    model,
    features: np.ndarray,
    targets: np.ndarray,
    known_kappa: np.ndarray,
    spec: OutputSpec,
    *,
    metadata: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate parameter, channel, observable, and correction metrics."""

    del metadata
    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    kappa = np.asarray(known_kappa, dtype=float)
    start = time.perf_counter()
    prediction = np.asarray(model.predict(x), dtype=float)
    elapsed = time.perf_counter() - start
    if prediction.shape != y.shape:
        raise ValueError("Model prediction has the wrong shape")

    per_parameter: dict[str, dict[str, float]] = {}
    r2_values = []
    for index, name in enumerate(spec.names):
        difference = prediction[:, index] - y[:, index]
        r2 = _r2_score(y[:, index], prediction[:, index])
        r2_values.append(r2)
        per_parameter[name] = {
            "mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "r2": r2,
        }

    protocol = (
        one_qubit_protocol()
        if spec.family == "1q"
        else two_qubit_protocol(
            gate_name="CX",
            basis=("ZI", "IZ", "ZX", "ZZ"),
            target_features=80,
        )
    )
    identity = identity_channel(1 if spec.family == "1q" else 2)
    rows: list[dict[str, Any]] = []
    correct_families = 0
    for sample in range(y.shape[0]):
        true_parameters = _parameters(spec, y[sample], kappa[sample])
        predicted_parameters = _parameters(spec, prediction[sample], kappa[sample])
        true_channel = build_channel(true_parameters)
        predicted_channel = build_channel(predicted_parameters)
        true_features = extract_features(
            build_diagnostic_channel(true_parameters), protocol
        )
        predicted_features = extract_features(
            build_diagnostic_channel(predicted_parameters), protocol
        )

        compensated_values = y[sample].copy()
        compensated_values[: spec.n_alpha] -= prediction[sample, : spec.n_alpha]
        compensated_parameters = _parameters(spec, compensated_values, kappa[sample])
        compensated_channel = build_channel(compensated_parameters)
        true_family = _dominant_family(spec, y[sample], kappa[sample])
        predicted_family = _dominant_family(spec, prediction[sample], kappa[sample])
        correct_families += int(true_family == predicted_family)
        rows.append(
            {
                "sample": sample,
                "choi_fidelity": choi_state_fidelity(true_channel, predicted_channel),
                "ptm_error": relative_ptm_frobenius_error(
                    true_channel, predicted_channel
                ),
                "heldout_expectation_rmse": float(
                    np.sqrt(np.mean((true_features - predicted_features) ** 2))
                ),
                "fidelity_before_compensation": choi_state_fidelity(
                    identity, true_channel
                ),
                "fidelity_after_compensation": choi_state_fidelity(
                    identity, compensated_channel
                ),
                "remaining_stochastic_error": float(
                    np.sum(
                        compensated_values[spec.n_alpha : spec.n_alpha + spec.n_gamma]
                    )
                    + (
                        compensated_values[-1]
                        if spec.has_kappa
                        else np.sum(kappa[sample])
                    )
                ),
                "strategy": (
                    "coherent_inverse"
                    if np.linalg.norm(prediction[sample, : spec.n_alpha]) > 1e-8
                    else "no_coherent_update"
                ),
                "true_family": true_family,
                "predicted_family": predicted_family,
            }
        )

    difference = prediction - y
    summary = {
        "parameter_mae": float(np.mean(np.abs(difference))),
        "parameter_rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_r2": float(np.mean(r2_values)),
        "choi_fidelity": float(np.mean([row["choi_fidelity"] for row in rows])),
        "ptm_error": float(np.mean([row["ptm_error"] for row in rows])),
        "heldout_expectation_rmse": float(
            np.mean([row["heldout_expectation_rmse"] for row in rows])
        ),
        "channel_family_accuracy": correct_families / max(len(rows), 1),
        "coherent_compensation_improvement": float(
            np.mean(
                [
                    row["fidelity_after_compensation"]
                    - row["fidelity_before_compensation"]
                    for row in rows
                ]
            )
        ),
        "inference_seconds_per_sample": elapsed / max(len(x), 1),
        "channel_samples_evaluated": len(rows),
    }
    return EvaluationResult(
        predictions=prediction,
        summary=summary,
        per_parameter=per_parameter,
        per_sample=rows,
    )
