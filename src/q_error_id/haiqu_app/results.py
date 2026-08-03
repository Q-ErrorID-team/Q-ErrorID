"""Conversion of Haiqu/Qiskit results into Agent 1 feature vectors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit


def normalize_distribution(
    distribution: Mapping[str, int | float],
) -> dict[str, float]:
    """Normalize counts or quasi-probabilities while preserving bitstrings."""

    clean = {
        str(key).replace(" ", ""): float(value) for key, value in distribution.items()
    }
    total = float(sum(clean.values()))
    if np.isclose(total, 0.0):
        raise ValueError("Distribution has zero total weight")
    return {key: value / total for key, value in clean.items()}


def pauli_expectation_from_counts(
    counts: Mapping[str, int | float],
    measurement_basis: str,
    circuit_qubits: Sequence[int] | None = None,
) -> float:
    """Compute a Pauli expectation from Qiskit's reversed-order bitstrings.

    ``measurement_basis`` follows Agent 1's qubit order. For example ``"ZX"``
    pairs with ``circuit_qubits=(0, 1)``. Basis rotations are already present in
    the diagnostic circuit.
    """

    probabilities = normalize_distribution(counts)
    basis = measurement_basis.replace(",", "")
    if circuit_qubits is None:
        circuit_qubits = tuple(range(len(basis)))
    if len(circuit_qubits) != len(basis):
        raise ValueError("circuit_qubits and measurement_basis have different lengths")

    expectation = 0.0
    required_width = max(circuit_qubits, default=-1) + 1
    for raw_bitstring, probability in probabilities.items():
        bitstring = raw_bitstring.zfill(required_width)
        parity = 0
        for pauli, qubit in zip(basis, circuit_qubits):
            if pauli == "I":
                continue
            if pauli not in "XYZ":
                raise ValueError(f"Invalid Pauli letter: {pauli}")
            offset = len(bitstring) - 1 - int(qubit)
            if offset < 0:
                raise ValueError("Bitstring is shorter than the measured qubit index")
            parity ^= int(bitstring[offset])
        expectation += (-1.0 if parity else 1.0) * probability
    return float(np.clip(expectation, -1.0, 1.0))


def _coerce_distributions(raw_results: Any) -> list[Mapping[str, int | float]]:
    if isinstance(raw_results, Mapping):
        return [raw_results]
    if not isinstance(raw_results, Sequence) or isinstance(raw_results, (str, bytes)):
        raise TypeError("Expected a distribution or a sequence of distributions")
    values = list(raw_results)
    while (
        len(values) == 1
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], (str, bytes, Mapping))
    ):
        values = list(values[0])
    if not all(isinstance(value, Mapping) for value in values):
        raise TypeError("Execution results do not contain measurement distributions")
    return values


@dataclass
class FeatureBatch:
    """Feature vectors and a row-wise audit table."""

    features: dict[str, np.ndarray]
    table: pd.DataFrame


def results_to_features(
    circuits: Sequence[QuantumCircuit],
    raw_results: Any,
    *,
    mode: str,
    readout_calibration: Any | None = None,
) -> FeatureBatch:
    """Convert ordered results into raw or calibrated Agent 1 feature vectors."""

    distributions = _coerce_distributions(raw_results)
    if len(distributions) != len(circuits):
        raise ValueError(
            f"Received {len(distributions)} results for {len(circuits)} circuits"
        )
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    rows = []
    for circuit, distribution in zip(circuits, distributions):
        metadata = circuit.metadata or {}
        basis = str(metadata["measurement_basis"])
        circuit_qubits = tuple(int(q) for q in metadata["circuit_qubits"])
        distribution_used: Mapping[str, int | float] = distribution
        expectation_qubits = circuit_qubits
        readout_audit: dict[str, float] = {}
        if readout_calibration is not None:
            distribution_used, readout_audit = (
                readout_calibration.correct_for_circuit(circuit, distribution)
            )
            expectation_qubits = tuple(range(len(basis.replace(",", ""))))
        value = pauli_expectation_from_counts(
            distribution_used,
            basis,
            expectation_qubits,
        )
        channel_key = str(metadata["channel_key"])
        feature_index = int(metadata["feature_index"])
        grouped[channel_key].append((feature_index, value))
        rows.append(
            {
                "mode": mode,
                "channel_key": channel_key,
                "feature_index": feature_index,
                "feature_label": metadata.get("feature_label"),
                "expectation": value,
                "shots_or_weight": float(sum(distribution.values())),
                "distribution": dict(distribution_used),
                "raw_distribution": dict(distribution),
                "readout_corrected": readout_calibration is not None,
                **readout_audit,
            }
        )

    features: dict[str, np.ndarray] = {}
    for channel_key, values in grouped.items():
        ordered = sorted(values)
        indices = [index for index, _ in ordered]
        if indices != list(range(len(indices))):
            raise ValueError(f"Non-contiguous feature indices for {channel_key}")
        features[channel_key] = np.asarray([value for _, value in ordered])
    return FeatureBatch(features=features, table=pd.DataFrame(rows))


def total_variation_distance(
    left: Mapping[str, int | float],
    right: Mapping[str, int | float],
) -> float:
    p = normalize_distribution(left)
    q = normalize_distribution(right)
    keys = p.keys() | q.keys()
    return 0.5 * float(sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in keys))


def success_probability(
    distribution: Mapping[str, int | float],
    success_states: Iterable[str],
) -> float:
    probabilities = normalize_distribution(distribution)
    return float(sum(probabilities.get(state, 0.0) for state in success_states))


def z_observable(
    distribution: Mapping[str, int | float],
    qubits: Sequence[int],
) -> float:
    width = max((len(str(key).replace(" ", "")) for key in distribution), default=0)
    basis = ["I"] * width
    for qubit in qubits:
        basis[int(qubit)] = "Z"
    active = tuple(range(width))
    return pauli_expectation_from_counts(distribution, "".join(basis), active)
