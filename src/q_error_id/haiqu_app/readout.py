"""Independent readout calibration and regularized assignment correction.

The calibration circuits are executed without Haiqu-managed mitigation.  Their
assignment matrices therefore describe the measured nuisance channel directly.
The raw distributions are always retained; this module only creates a separate
readout-corrected view for generator reconstruction and algorithm validation.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit


def ordered_bitstrings(width: int) -> tuple[str, ...]:
    """Return Qiskit display-order bitstrings in numeric order."""

    if width < 1:
        raise ValueError("width must be positive")
    return tuple(format(index, f"0{width}b") for index in range(2**width))


def _normalize(distribution: Mapping[str, int | float]) -> dict[str, float]:
    clean = {
        str(key).replace(" ", ""): float(value)
        for key, value in distribution.items()
    }
    total = float(sum(clean.values()))
    if np.isclose(total, 0.0):
        raise ValueError("Distribution has zero total weight")
    return {key: value / total for key, value in clean.items()}


def marginalize_distribution(
    distribution: Mapping[str, int | float],
    measurement_clbits: Sequence[int],
) -> dict[str, float]:
    """Marginalize counts onto logical clbits in Qiskit display order.

    ``measurement_clbits=(c0, c1)`` returns strings ordered as ``c1 c0``.
    This matches the count convention used by the Grover targets.
    """

    clbits = tuple(int(bit) for bit in measurement_clbits)
    if not clbits or len(set(clbits)) != len(clbits) or min(clbits) < 0:
        raise ValueError("measurement_clbits must contain distinct nonnegative bits")
    states = ordered_bitstrings(len(clbits))
    output = dict.fromkeys(states, 0.0)
    for raw_bitstring, probability in _normalize(distribution).items():
        if max(clbits) >= len(raw_bitstring):
            raise ValueError(
                "A measured bitstring is shorter than the requested classical bit"
            )
        active = "".join(
            raw_bitstring[len(raw_bitstring) - 1 - clbit]
            for clbit in reversed(clbits)
        )
        output[active] += probability
    return {state: float(output[state]) for state in states}


def _project_probability_simplex(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("A finite nonempty vector is required")
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1)
    positive = ordered - cumulative / indices > 0.0
    rho = int(np.flatnonzero(positive)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    return projected / projected.sum()


@dataclass(frozen=True)
class ReadoutAssignment:
    """Measured assignment matrix and its regularized non-CPTP inverse."""

    key: str
    physical_qubits: tuple[int, ...]
    bitstrings: tuple[str, ...]
    assignment_matrix: np.ndarray
    inverse_matrix: np.ndarray
    regularization: float
    condition_number: float
    mean_assignment_fidelity: float
    inverse_overhead_l1: float
    validation_passed: bool

    @classmethod
    def from_matrix(
        cls,
        *,
        key: str,
        physical_qubits: Sequence[int],
        matrix: np.ndarray,
        regularization: float = 1e-6,
    ) -> ReadoutAssignment:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("The readout assignment matrix must be square")
        width = round(np.log2(matrix.shape[0]))
        if 2**width != matrix.shape[0]:
            raise ValueError("The assignment dimension must be a power of two")
        qubits = tuple(int(qubit) for qubit in physical_qubits)
        if len(qubits) != width:
            raise ValueError("physical_qubits and assignment width disagree")
        if regularization < 0.0:
            raise ValueError("regularization must be nonnegative")
        if np.any(matrix < -1e-12):
            raise ValueError("Assignment probabilities must be nonnegative")
        column_sums = matrix.sum(axis=0)
        if np.any(column_sums <= 0.0):
            raise ValueError("Every prepared calibration state needs observations")
        normalized = matrix / column_sums
        u_matrix, singular_values, vh_matrix = np.linalg.svd(
            normalized,
            full_matrices=False,
        )
        factors = singular_values / (
            np.square(singular_values) + float(regularization)
        )
        inverse = (vh_matrix.T * factors) @ u_matrix.T
        condition = float(np.linalg.cond(normalized))
        mean_fidelity = float(np.mean(np.diag(normalized)))
        validation = bool(
            np.all(np.isfinite(normalized))
            and np.isfinite(condition)
            and condition < 50.0
            and mean_fidelity > 0.5
            and np.allclose(normalized.sum(axis=0), 1.0, atol=1e-9)
        )
        return cls(
            key=str(key),
            physical_qubits=qubits,
            bitstrings=ordered_bitstrings(width),
            assignment_matrix=normalized,
            inverse_matrix=inverse,
            regularization=float(regularization),
            condition_number=condition,
            mean_assignment_fidelity=mean_fidelity,
            inverse_overhead_l1=float(np.linalg.norm(inverse, ord=1)),
            validation_passed=validation,
        )

    @classmethod
    def identity(
        cls,
        *,
        key: str,
        physical_qubits: Sequence[int],
    ) -> ReadoutAssignment:
        qubits = tuple(int(qubit) for qubit in physical_qubits)
        return cls.from_matrix(
            key=key,
            physical_qubits=qubits,
            matrix=np.eye(2 ** len(qubits)),
            regularization=0.0,
        )

    def correct_with_audit(
        self,
        distribution: Mapping[str, int | float],
        *,
        require_validation: bool = True,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Invert the assignment channel and report projection diagnostics."""

        if require_validation and not self.validation_passed:
            raise RuntimeError(f"Readout calibration {self.key} failed validation")
        probabilities = _normalize(distribution)
        observed = np.asarray(
            [probabilities.get(state, 0.0) for state in self.bitstrings],
            dtype=float,
        )
        observed /= observed.sum()
        quasiprobabilities = self.inverse_matrix @ observed
        corrected = _project_probability_simplex(quasiprobabilities)
        output = {
            state: float(corrected[index])
            for index, state in enumerate(self.bitstrings)
        }
        audit = {
            "readout_raw_normalization": float(quasiprobabilities.sum()),
            "readout_raw_negativity": float(
                np.abs(quasiprobabilities[quasiprobabilities < 0.0]).sum()
            ),
            "readout_simplex_projection_l1": float(
                np.linalg.norm(corrected - quasiprobabilities, ord=1)
            ),
            "readout_condition_number": self.condition_number,
            "readout_inverse_overhead_l1": self.inverse_overhead_l1,
        }
        return output, audit

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "physical_qubits": list(self.physical_qubits),
            "bitstring_order": list(self.bitstrings),
            "assignment_matrix": self.assignment_matrix.tolist(),
            "regularized_inverse_matrix": self.inverse_matrix.tolist(),
            "regularization": self.regularization,
            "condition_number": self.condition_number,
            "mean_assignment_fidelity": self.mean_assignment_fidelity,
            "inverse_overhead_l1": self.inverse_overhead_l1,
            "validation_passed": self.validation_passed,
        }


@dataclass
class ReadoutCalibrationBundle:
    """One- and two-qubit assignment matrices for the selected subgraph."""

    single_qubit: dict[int, ReadoutAssignment]
    two_qubit: dict[str, ReadoutAssignment]
    calibration_shots: int
    regularization: float

    @classmethod
    def from_results(
        cls,
        circuits: Sequence[QuantumCircuit],
        distributions: Sequence[Mapping[str, int | float]],
        *,
        regularization: float = 1e-6,
        expected_shots: int | None = None,
    ) -> ReadoutCalibrationBundle:
        if len(circuits) != len(distributions):
            raise ValueError("Calibration circuits and results have different lengths")
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "physical_qubits": None,
                "bitstrings": None,
                "columns": {},
            }
        )
        shot_counts = []
        for circuit, distribution in zip(circuits, distributions):
            metadata = circuit.metadata or {}
            if metadata.get("calibration_type") != "readout_assignment":
                raise ValueError("A non-readout circuit was passed as calibration")
            key = str(metadata["calibration_key"])
            prepared = str(metadata["prepared_state"])
            clbits = tuple(int(bit) for bit in metadata["measurement_clbits"])
            bitstrings = tuple(str(state) for state in metadata["bitstring_order"])
            marginal = marginalize_distribution(distribution, clbits)
            group = grouped[key]
            group["physical_qubits"] = tuple(
                int(qubit) for qubit in metadata["physical_qubits"]
            )
            group["bitstrings"] = bitstrings
            group["columns"][prepared] = marginal
            total_weight = float(sum(float(x) for x in distribution.values()))
            if expected_shots is not None:
                if expected_shots < 1:
                    raise ValueError("expected_shots must be positive")
                shot_counts.append(int(expected_shots))
            elif np.isclose(total_weight, 1.0):
                # A probability or quasi-probability payload does not contain
                # enough information to reconstruct its original shot count.
                shot_counts.append(0)
            else:
                shot_counts.append(round(total_weight))

        singles: dict[int, ReadoutAssignment] = {}
        pairs: dict[str, ReadoutAssignment] = {}
        for key, group in grouped.items():
            bitstrings = tuple(group["bitstrings"])
            if set(group["columns"]) != set(bitstrings):
                missing = sorted(set(bitstrings) - set(group["columns"]))
                raise ValueError(f"Calibration {key} is missing prepared states {missing}")
            matrix = np.column_stack(
                [
                    np.asarray(
                        [group["columns"][prepared][measured] for measured in bitstrings],
                        dtype=float,
                    )
                    for prepared in bitstrings
                ]
            )
            assignment = ReadoutAssignment.from_matrix(
                key=key,
                physical_qubits=group["physical_qubits"],
                matrix=matrix,
                regularization=regularization,
            )
            if len(assignment.physical_qubits) == 1:
                singles[assignment.physical_qubits[0]] = assignment
            elif len(assignment.physical_qubits) == 2:
                pairs[key] = assignment
            else:
                raise ValueError("Only 1Q and 2Q readout calibrations are supported")
        return cls(
            single_qubit=singles,
            two_qubit=pairs,
            calibration_shots=min(shot_counts, default=0),
            regularization=float(regularization),
        )

    @classmethod
    def identity(
        cls,
        qubits: Sequence[int],
        edges: Sequence[Sequence[int]],
    ) -> ReadoutCalibrationBundle:
        singles = {
            int(qubit): ReadoutAssignment.identity(
                key=f"q{int(qubit)}",
                physical_qubits=(int(qubit),),
            )
            for qubit in qubits
        }
        pairs = {
            f"q{int(left)}-q{int(right)}": ReadoutAssignment.identity(
                key=f"q{int(left)}-q{int(right)}",
                physical_qubits=(int(left), int(right)),
            )
            for left, right in edges
        }
        return cls(
            single_qubit=singles,
            two_qubit=pairs,
            calibration_shots=0,
            regularization=0.0,
        )

    def assignment_for_key(self, channel_key: str) -> ReadoutAssignment:
        if "-" in channel_key:
            try:
                return self.two_qubit[channel_key]
            except KeyError as exc:
                raise KeyError(f"No readout calibration exists for {channel_key}") from exc
        qubit = int(channel_key.removeprefix("q"))
        try:
            return self.single_qubit[qubit]
        except KeyError as exc:
            raise KeyError(f"No readout calibration exists for q{qubit}") from exc

    def correct_for_circuit(
        self,
        circuit: QuantumCircuit,
        distribution: Mapping[str, int | float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        metadata = circuit.metadata or {}
        channel_key = str(metadata["channel_key"])
        assignment = self.assignment_for_key(channel_key)
        clbits = tuple(
            int(bit)
            for bit in metadata.get(
                "measurement_clbits",
                metadata.get("circuit_qubits", ()),
            )
        )
        marginal = marginalize_distribution(distribution, clbits)
        return assignment.correct_with_audit(marginal)

    def correct_edge(
        self,
        edge_key: str,
        distribution: Mapping[str, int | float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        assignment = self.assignment_for_key(edge_key)
        marginal = marginalize_distribution(
            distribution,
            tuple(range(len(assignment.physical_qubits))),
        )
        return assignment.correct_with_audit(marginal)

    def independent_assignment(
        self,
        physical_qubits: Sequence[int],
    ) -> ReadoutAssignment:
        """Compose calibrated 1Q matrices without assuming correlated readout."""

        qubits = tuple(int(qubit) for qubit in physical_qubits)
        states = ordered_bitstrings(len(qubits))
        matrix = np.zeros((len(states), len(states)), dtype=float)
        for column, prepared in enumerate(states):
            for row, measured in enumerate(states):
                probability = 1.0
                for logical_index, physical in enumerate(qubits):
                    one = self.single_qubit[physical]
                    prepared_bit = int(prepared[-1 - logical_index])
                    measured_bit = int(measured[-1 - logical_index])
                    probability *= one.assignment_matrix[measured_bit, prepared_bit]
                matrix[row, column] = probability
        return ReadoutAssignment.from_matrix(
            key="independent:" + ",".join(f"q{qubit}" for qubit in qubits),
            physical_qubits=qubits,
            matrix=matrix,
            regularization=self.regularization,
        )

    def correct_joint(
        self,
        distribution: Mapping[str, int | float],
        physical_qubits: Sequence[int],
    ) -> tuple[dict[str, float], dict[str, float]]:
        assignment = self.independent_assignment(physical_qubits)
        marginal = marginalize_distribution(
            distribution,
            tuple(range(len(assignment.physical_qubits))),
        )
        return assignment.correct_with_audit(marginal)

    @property
    def validation_passed(self) -> bool:
        assignments = [*self.single_qubit.values(), *self.two_qubit.values()]
        return bool(assignments) and all(item.validation_passed for item in assignments)

    def dataframe(self) -> pd.DataFrame:
        rows = []
        for assignment_type, assignments in (
            ("single_qubit", self.single_qubit.values()),
            ("two_qubit", self.two_qubit.values()),
        ):
            for assignment in assignments:
                for prepared_index, prepared in enumerate(assignment.bitstrings):
                    for measured_index, measured in enumerate(assignment.bitstrings):
                        rows.append(
                            {
                                "assignment_type": assignment_type,
                                "calibration_key": assignment.key,
                                "physical_qubits": list(assignment.physical_qubits),
                                "prepared_state": prepared,
                                "measured_state": measured,
                                "probability": assignment.assignment_matrix[
                                    measured_index,
                                    prepared_index,
                                ],
                                "condition_number": assignment.condition_number,
                                "mean_assignment_fidelity": (
                                    assignment.mean_assignment_fidelity
                                ),
                                "inverse_overhead_l1": (
                                    assignment.inverse_overhead_l1
                                ),
                                "validation_passed": assignment.validation_passed,
                                "shots": self.calibration_shots,
                            }
                        )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "semantics": (
                "independently measured assignment correction; raw counts retained; "
                "regularized inverse projected to the probability simplex"
            ),
            "calibration_shots": self.calibration_shots,
            "regularization": self.regularization,
            "validation_passed": self.validation_passed,
            "single_qubit": {
                f"q{qubit}": assignment.to_dict()
                for qubit, assignment in self.single_qubit.items()
            },
            "two_qubit": {
                key: assignment.to_dict()
                for key, assignment in self.two_qubit.items()
            },
        }


__all__ = [
    "ReadoutAssignment",
    "ReadoutCalibrationBundle",
    "marginalize_distribution",
    "ordered_bitstrings",
]
