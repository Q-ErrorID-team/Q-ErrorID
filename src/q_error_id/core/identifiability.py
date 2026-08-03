"""Numerical identifiability analysis and feature-bank selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import qr

from .channels import build_diagnostic_channel
from .parameters import (
    ChannelParameters,
    one_qubit_parameters,
    parameter_bounds,
    two_qubit_parameters,
)
from .protocols import PrepareMeasureProtocol, extract_features


@dataclass(frozen=True)
class IdentifiabilityResult:
    """Structured summary of a feature-map Jacobian."""

    rank: int
    parameter_count: int
    feature_count: int
    condition_number: float
    singular_values: tuple[float, ...]
    minimal_reliable_indices: tuple[int, ...]
    minimal_reliable_features: tuple[str, ...]
    redundant_indices: tuple[int, ...]
    redundant_features: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report."""

        return {
            "rank": self.rank,
            "parameter_count": self.parameter_count,
            "feature_count": self.feature_count,
            "condition_number": self.condition_number,
            "singular_values": list(self.singular_values),
            "minimal_reliable_indices": list(self.minimal_reliable_indices),
            "minimal_reliable_features": list(self.minimal_reliable_features),
            "redundant_indices": list(self.redundant_indices),
            "redundant_features": list(self.redundant_features),
        }


def representative_parameters(
    gate_name: str = "1Q",
    basis: tuple[str, ...] | None = None,
) -> ChannelParameters:
    """Return a well-interior point used for local Jacobian design."""

    if gate_name.upper() in {"1Q", "ONE_QUBIT"}:
        return one_qubit_parameters(
            alpha=np.array([0.045, -0.032, 0.026]),
            gamma=np.array([0.010, 0.013, 0.008]),
            kappa_down=0.021,
        )
    return two_qubit_parameters(
        gate_name=gate_name,
        basis=basis,
        alpha=np.array([0.043, -0.031, 0.027, 0.019]),
        gamma=np.array([0.010, 0.013, 0.008, 0.016]),
        kappa_down=np.array([0.019, 0.023]),
    )


def numerical_feature_jacobian(
    parameters: ChannelParameters,
    protocol: PrepareMeasureProtocol,
    *,
    include_kappa: bool = True,
    relative_step: float = 2e-5,
) -> np.ndarray:
    """Differentiate the exact feature map by bounded finite differences."""

    center = parameters.as_vector(include_kappa=include_kappa)
    lower, upper = parameter_bounds(parameters, include_kappa=include_kappa)
    baseline = extract_features(build_diagnostic_channel(parameters), protocol)
    jacobian = np.empty((baseline.size, center.size), dtype=float)

    for column in range(center.size):
        step = relative_step * max(1.0, abs(center[column]))
        forward = center.copy()
        backward = center.copy()
        can_forward = center[column] + step <= upper[column]
        can_backward = center[column] - step >= lower[column]
        if can_forward and can_backward:
            forward[column] += step
            backward[column] -= step
            plus = extract_features(
                build_diagnostic_channel(
                    parameters.with_vector(forward, include_kappa=include_kappa)
                ),
                protocol,
            )
            minus = extract_features(
                build_diagnostic_channel(
                    parameters.with_vector(backward, include_kappa=include_kappa)
                ),
                protocol,
            )
            jacobian[:, column] = (plus - minus) / (2.0 * step)
        elif can_forward:
            forward[column] += step
            plus = extract_features(
                build_diagnostic_channel(
                    parameters.with_vector(forward, include_kappa=include_kappa)
                ),
                protocol,
            )
            jacobian[:, column] = (plus - baseline) / step
        else:
            backward[column] -= step
            minus = extract_features(
                build_diagnostic_channel(
                    parameters.with_vector(backward, include_kappa=include_kappa)
                ),
                protocol,
            )
            jacobian[:, column] = (baseline - minus) / step
    return jacobian


def analyze_jacobian(
    jacobian: np.ndarray,
    feature_labels: tuple[str, ...],
    *,
    relative_tolerance: float = 1e-8,
) -> IdentifiabilityResult:
    """Report rank, conditioning, independent rows, and redundant features."""

    matrix = np.asarray(jacobian, dtype=float)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = (
        0.0 if singular_values.size == 0 else relative_tolerance * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > threshold))
    if rank:
        condition_number = float(singular_values[0] / singular_values[rank - 1])
        _, _, pivots = qr(matrix.T, mode="economic", pivoting=True)
        independent = tuple(int(index) for index in pivots[:rank])
    else:
        condition_number = float("inf")
        independent = ()
    independent_set = set(independent)
    redundant = tuple(
        index for index in range(matrix.shape[0]) if index not in independent_set
    )
    return IdentifiabilityResult(
        rank=rank,
        parameter_count=matrix.shape[1],
        feature_count=matrix.shape[0],
        condition_number=condition_number,
        singular_values=tuple(float(value) for value in singular_values),
        minimal_reliable_indices=independent,
        minimal_reliable_features=tuple(feature_labels[index] for index in independent),
        redundant_indices=redundant,
        redundant_features=tuple(feature_labels[index] for index in redundant),
    )


def analyze_identifiability(
    parameters: ChannelParameters,
    protocol: PrepareMeasureProtocol,
    *,
    include_kappa: bool = True,
) -> IdentifiabilityResult:
    """Analyze the local feature map at the supplied physical parameters."""

    jacobian = numerical_feature_jacobian(
        parameters, protocol, include_kappa=include_kappa
    )
    return analyze_jacobian(jacobian, protocol.feature_labels)


def select_protocol(
    parameters: ChannelParameters,
    candidate_protocol: PrepareMeasureProtocol,
    *,
    target_features: int = 80,
    include_kappa: bool = True,
) -> tuple[PrepareMeasureProtocol, dict[str, object]]:
    """Select a full-rank bank and augment it with informative redundancy."""

    if target_features > candidate_protocol.feature_count:
        raise ValueError("target_features exceeds the candidate-bank size")
    jacobian = numerical_feature_jacobian(
        parameters, candidate_protocol, include_kappa=include_kappa
    )
    candidate_report = analyze_jacobian(jacobian, candidate_protocol.feature_labels)
    if candidate_report.rank < candidate_report.parameter_count:
        raise RuntimeError(
            "Candidate bank is not locally identifiable: "
            f"rank {candidate_report.rank}/{candidate_report.parameter_count}"
        )

    independent = list(candidate_report.minimal_reliable_indices)
    selected_set = set(independent)
    remaining = [
        index
        for index in range(candidate_protocol.feature_count)
        if index not in selected_set
    ]

    # Add high-sensitivity rows first.  The independent QR seed guarantees full
    # rank, while these rows improve shot-noise robustness without pretending
    # that every additional scalar is mathematically independent.
    norms = np.linalg.norm(jacobian, axis=1)
    remaining.sort(key=lambda index: (-norms[index], index))
    selected_indices = (
        independent + remaining[: max(0, target_features - len(independent))]
    )
    selected = candidate_protocol.subset(
        selected_indices,
        name=f"{parameters.gate_name.lower()}_auto_{target_features}_features",
    )
    selected_jacobian = jacobian[selected_indices]
    selected_report = analyze_jacobian(selected_jacobian, selected.feature_labels)
    minimal_jacobian = jacobian[independent]
    minimal_singular_values = np.linalg.svd(minimal_jacobian, compute_uv=False)
    minimal_condition = float(minimal_singular_values[0] / minimal_singular_values[-1])
    report: dict[str, object] = {
        "candidate_feature_count": candidate_protocol.feature_count,
        "selected_feature_count": selected.feature_count,
        "rank": selected_report.rank,
        "parameter_count": selected_report.parameter_count,
        "condition_number": selected_report.condition_number,
        "minimal_reliable_feature_count": len(independent),
        "minimal_reliable_condition_number": minimal_condition,
        "minimal_reliable_features": [
            candidate_protocol.feature_labels[index] for index in independent
        ],
        "redundant_selected_features": [
            selected.feature_labels[index]
            for index in range(len(independent), selected.feature_count)
        ],
        "selected_candidate_indices": selected_indices,
    }
    return selected, report
