"""Algorithm-level Grover benchmark and learned-generator error cancellation.

The primary benchmark propagates the complete reconstructed channel after
every modeled logical gate, builds the resulting algorithm response matrix,
and applies its regularized inverse in classical post-processing.  This uses
coherent, stochastic, and non-unital components without adding noisy physical
correction gates.

The module also exposes an explicit gate-local coherent inverse circuit for
controlled studies.  That circuit is useful for demonstrating placement
semantics, but its extra gates are not assumed to be free.  Neither path is
fault-tolerant quantum error correction: irreversible noise has no physical
CPTP inverse.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import expm

from q_error_id.core import (
    build_channel,
    channel_to_kraus,
    one_qubit_parameters,
    two_qubit_parameters,
)
from q_error_id.core.pauli import I, X, Y, Z, pauli_word

GROVER_TARGETS = ("00", "01", "10", "11")
ONE_Q_LABELS = ("X", "Y", "Z")
TWO_Q_LABELS = ("ZI", "IZ", "ZX", "ZZ")
_ONE_Q_MATRICES = {"X": X, "H": (X + Z) / np.sqrt(2.0)}


def _validated_target(target: str) -> str:
    normalized = str(target).replace(" ", "")
    if normalized not in GROVER_TARGETS:
        raise ValueError("The two-qubit Grover target must be 00, 01, 10, or 11")
    return normalized


def grover_operation_sequence(
    target: str,
    *,
    two_qubit_depth: int = 2,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return an ideal-preserving Grover depth-sweep sequence.

    Targets follow Qiskit's displayed count order ``c1 c0``.  The first
    logical qubit is therefore addressed by the rightmost target bit.
    The base algorithm contains two CX gates.  Larger requested depths append
    CX-CX identity folds, so the ideal output remains the requested target
    while device noise is allowed to accumulate.
    """

    target = _validated_target(target)
    if two_qubit_depth < 2 or two_qubit_depth % 2:
        raise ValueError("two_qubit_depth must be an even integer >= 2")
    logical_bits = (int(target[1]), int(target[0]))
    operations: list[tuple[str, tuple[int, ...]]] = [
        ("h", (0,)),
        ("h", (1,)),
    ]

    # Oracle: map the requested state to |11>, apply CZ = H-CX-H, and unmap.
    for qubit, bit in enumerate(logical_bits):
        if bit == 0:
            operations.append(("x", (qubit,)))
    operations.extend((("h", (1,)), ("cx", (0, 1)), ("h", (1,))))
    for qubit, bit in reversed(tuple(enumerate(logical_bits))):
        if bit == 0:
            operations.append(("x", (qubit,)))

    # Standard two-qubit diffusion operator, again decomposing CZ through CX.
    operations.extend(
        (
            ("h", (0,)),
            ("h", (1,)),
            ("x", (0,)),
            ("x", (1,)),
            ("h", (1,)),
            ("cx", (0, 1)),
            ("h", (1,)),
            ("x", (0,)),
            ("x", (1,)),
            ("h", (0,)),
            ("h", (1,)),
        )
    )
    operations.extend(
        ("cx", (0, 1)) for _ in range(two_qubit_depth - 2)
    )
    return tuple(operations)


def _channel_values(channel: Mapping[str, Any], field: str, labels: Sequence[str]):
    values = channel.get(field, {})
    return np.asarray([float(values.get(label, 0.0)) for label in labels], dtype=float)


def _single_kappa(channel: Mapping[str, Any]) -> float:
    values = channel.get("kappa_down", {})
    if isinstance(values, Mapping):
        return float(max((float(value) for value in values.values()), default=0.0))
    array = np.asarray(values, dtype=float).reshape(-1)
    return float(array[0]) if array.size else 0.0


def _coherent_inverse_1q(channel: Mapping[str, Any]) -> np.ndarray:
    alpha = _channel_values(channel, "alpha", ONE_Q_LABELS)
    hamiltonian = 0.5 * (alpha[0] * X + alpha[1] * Y + alpha[2] * Z)
    return expm(1.0j * hamiltonian)


def _coherent_inverse_2q(
    channel: Mapping[str, Any],
    *,
    qiskit_order: bool,
) -> np.ndarray:
    alpha = _channel_values(channel, "alpha", TWO_Q_LABELS)
    hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    for coefficient, label in zip(alpha, TWO_Q_LABELS):
        # Core channel labels use first-listed-qubit tensor order.  Qiskit
        # matrices use little-endian qargs order, hence the explicit reversal.
        matrix_label = label[::-1] if qiskit_order else label
        hamiltonian += 0.5 * coefficient * pauli_word(matrix_label)
    return expm(1.0j * hamiltonian)


def _nonzero_coherent(channel: Mapping[str, Any], labels: Sequence[str]) -> bool:
    return bool(np.linalg.norm(_channel_values(channel, "alpha", labels)) > 1e-14)


def build_grover_search_circuit(
    target: str,
    *,
    width: int = 4,
    circuit_qubits: tuple[int, int] = (0, 1),
    physical_qubits: tuple[int, int] | None = None,
    edge_key: str | None = None,
    single_qubit_channels: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
    two_qubit_channel: Mapping[str, Any] | None = None,
    two_qubit_depth: int = 2,
) -> QuantumCircuit:
    """Build a measured Grover circuit with optional gate-local compensation."""

    target = _validated_target(target)
    q0, q1 = (int(circuit_qubits[0]), int(circuit_qubits[1]))
    if q0 == q1 or min(q0, q1) < 0 or max(q0, q1) >= width:
        raise ValueError("circuit_qubits must identify two distinct qubits in width")
    physical = tuple(int(q) for q in (physical_qubits or circuit_qubits))
    if len(physical) != 2:
        raise ValueError("physical_qubits must contain two entries")

    correction_requested = (
        single_qubit_channels is not None or two_qubit_channel is not None
    )
    if correction_requested and (
        single_qubit_channels is None or two_qubit_channel is None
    ):
        raise ValueError(
            "Both one-qubit channels and the two-qubit channel are required "
            "for gate-local correction"
        )

    circuit = QuantumCircuit(width, 2)
    logical_to_circuit = {0: q0, 1: q1}
    correction_locations = 0
    logical_gate_count = 0
    logical_two_qubit_gate_count = 0

    base_operation_count = len(grover_operation_sequence(target))
    for operation_index, (gate_name, logical_qubits) in enumerate(
        grover_operation_sequence(
            target,
            two_qubit_depth=two_qubit_depth,
        )
    ):
        if operation_index >= base_operation_count:
            # Barriers keep the deliberately inserted CX-CX identity folds
            # visible to optimization-level 2 transpilation.
            circuit.barrier(q0, q1)
        mapped = tuple(logical_to_circuit[q] for q in logical_qubits)
        if gate_name == "h":
            circuit.h(mapped[0])
        elif gate_name == "x":
            circuit.x(mapped[0])
        elif gate_name == "cx":
            circuit.cx(mapped[0], mapped[1])
            logical_two_qubit_gate_count += 1
        else:  # pragma: no cover - the sequence above is a closed contract
            raise RuntimeError(f"Unsupported Grover operation: {gate_name}")
        logical_gate_count += 1

        if not correction_requested:
            continue
        if len(logical_qubits) == 1:
            channel = single_qubit_channels[logical_qubits[0]]
            if _nonzero_coherent(channel, ONE_Q_LABELS):
                circuit.unitary(
                    _coherent_inverse_1q(channel),
                    [mapped[0]],
                    label=f"inv_L1_q{logical_qubits[0]}",
                )
                correction_locations += 1
        else:
            if _nonzero_coherent(two_qubit_channel, TWO_Q_LABELS):
                circuit.unitary(
                    _coherent_inverse_2q(
                        two_qubit_channel,
                        qiskit_order=True,
                    ),
                    list(mapped),
                    label="inv_L2_edge",
                )
                correction_locations += 1

    circuit.barrier()
    circuit.measure(q0, 0)
    circuit.measure(q1, 1)
    suffix = "generator_corrected" if correction_requested else "raw"
    compact_edge = edge_key or f"q{physical[0]}-q{physical[1]}"
    circuit.name = (
        f"grover_{target}_{compact_edge}_d2q{two_qubit_depth}_{suffix}"
    )
    circuit.metadata = {
        "algorithm": "grover_2q_exhaustive",
        "target_state": target,
        "known_success_states": [target],
        "edge_key": compact_edge,
        "gate_name": "grover_oracle_and_diffusion",
        "physical_qubits": list(physical),
        "circuit_qubits": [q0, q1],
        "measurement_clbits": [0, 1],
        "logical_gate_count": logical_gate_count,
        "logical_two_qubit_gate_count": logical_two_qubit_gate_count,
        "requested_two_qubit_depth": int(two_qubit_depth),
        "depth_sweep_method": "cx_cx_identity_folding",
        "coherent_correction": correction_requested,
        "coherent_correction_locations": correction_locations,
        "correction_at_end_only": False,
        "correction_semantics": (
            "gate_local_exact_inverse_of_reconstructed_coherent_generator"
            if correction_requested
            else "none"
        ),
    }
    return circuit


def _unitary_evolution(density_matrix: np.ndarray, unitary: np.ndarray) -> np.ndarray:
    return unitary @ density_matrix @ unitary.conj().T


def _embedded_one_qubit_unitary(unitary: np.ndarray, qubit: int) -> np.ndarray:
    if qubit == 0:
        return np.kron(unitary, I)
    if qubit == 1:
        return np.kron(I, unitary)
    raise ValueError("The forward model contains exactly two logical qubits")


def _apply_local_channel(
    density_matrix: np.ndarray,
    kraus_operators: Sequence[np.ndarray],
    qubit: int,
) -> np.ndarray:
    output = np.zeros_like(density_matrix)
    for operator in kraus_operators:
        embedded = (
            np.kron(operator, I) if qubit == 0 else np.kron(I, operator)
        )
        output += embedded @ density_matrix @ embedded.conj().T
    return output


def _density_to_qiskit_distribution(
    density_matrix: np.ndarray,
) -> dict[str, float]:
    probabilities = np.real_if_close(np.diag(density_matrix)).astype(float)
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    # The core simulator orders basis states as |q0 q1>; Qiskit displays c1 c0.
    return {
        qiskit_bits: float(probabilities[index])
        for index, qiskit_bits in enumerate(("00", "10", "01", "11"))
    }


def simulate_grover_with_generator(
    target: str,
    *,
    single_qubit_channels: tuple[Mapping[str, Any], Mapping[str, Any]],
    two_qubit_channel: Mapping[str, Any],
    coherent_compensation: bool,
    two_qubit_depth: int = 2,
) -> dict[str, float]:
    """Forward-simulate the Grover circuit under the reconstructed generator."""

    one_kraus = []
    one_inverse = []
    for logical_qubit, channel in enumerate(single_qubit_channels):
        parameters = one_qubit_parameters(
            alpha=_channel_values(channel, "alpha", ONE_Q_LABELS),
            gamma=_channel_values(channel, "gamma", ONE_Q_LABELS),
            kappa_down=_single_kappa(channel),
            qubit=logical_qubit,
        )
        core_channel = build_channel(parameters)
        one_kraus.append(channel_to_kraus(core_channel))
        one_inverse.append(_coherent_inverse_1q(channel))

    two_parameters = two_qubit_parameters(
        gate_name="CX",
        alpha=_channel_values(two_qubit_channel, "alpha", TWO_Q_LABELS),
        gamma=_channel_values(two_qubit_channel, "gamma", TWO_Q_LABELS),
        kappa_down=np.asarray(
            [_single_kappa(channel) for channel in single_qubit_channels],
            dtype=float,
        ),
        qubits=(0, 1),
        basis=TWO_Q_LABELS,
    )
    two_channel = build_channel(two_parameters)
    two_inverse = _coherent_inverse_2q(
        two_qubit_channel,
        qiskit_order=False,
    )

    density_matrix = np.zeros((4, 4), dtype=np.complex128)
    density_matrix[0, 0] = 1.0
    for gate_name, logical_qubits in grover_operation_sequence(
        target,
        two_qubit_depth=two_qubit_depth,
    ):
        if gate_name in {"h", "x"}:
            qubit = logical_qubits[0]
            ideal = _embedded_one_qubit_unitary(
                _ONE_Q_MATRICES[gate_name.upper()],
                qubit,
            )
            density_matrix = _unitary_evolution(density_matrix, ideal)
            density_matrix = _apply_local_channel(
                density_matrix,
                one_kraus[qubit],
                qubit,
            )
            if coherent_compensation:
                correction = _embedded_one_qubit_unitary(
                    one_inverse[qubit],
                    qubit,
                )
                density_matrix = _unitary_evolution(density_matrix, correction)
        elif gate_name == "cx":
            from q_error_id.core.channels import IDEAL_CX

            density_matrix = _unitary_evolution(density_matrix, IDEAL_CX)
            density_matrix = two_channel.apply(density_matrix)
            if coherent_compensation:
                density_matrix = _unitary_evolution(
                    density_matrix,
                    two_inverse,
                )
        else:  # pragma: no cover - the operation contract is closed
            raise RuntimeError(f"Unsupported Grover operation: {gate_name}")

    density_matrix = (density_matrix + density_matrix.conj().T) / 2.0
    trace = float(np.real_if_close(np.trace(density_matrix)))
    if trace <= 0.0:
        raise RuntimeError("The learned forward model produced a nonpositive trace")
    return _density_to_qiskit_distribution(density_matrix / trace)


def distribution_vector(
    distribution: Mapping[str, int | float],
    *,
    bitstrings: Sequence[str] = GROVER_TARGETS,
) -> np.ndarray:
    """Convert counts or quasi-probabilities to a fixed ordered vector."""

    cleaned = {
        str(key).replace(" ", ""): float(value)
        for key, value in distribution.items()
    }
    vector = np.asarray([cleaned.get(key, 0.0) for key in bitstrings], dtype=float)
    total = float(vector.sum())
    if np.isclose(total, 0.0):
        raise ValueError("Distribution has zero total weight")
    return vector / total


def project_probability_simplex(values: Sequence[float]) -> np.ndarray:
    """Euclidean projection onto ``p >= 0`` and ``sum(p) = 1``."""

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


def hellinger_fidelity(
    left: Mapping[str, int | float],
    right: Mapping[str, int | float],
) -> float:
    """Return the squared Bhattacharyya coefficient of two distributions."""

    p = project_probability_simplex(distribution_vector(left))
    q = project_probability_simplex(distribution_vector(right))
    return float(np.square(np.sum(np.sqrt(p * q))))


@dataclass(frozen=True)
class GeneratorResponseModel:
    """Regularized algorithm response inverse built only from the error atlas."""

    edge_key: str
    response_matrix: np.ndarray
    inverse_matrix: np.ndarray
    regularization: float
    condition_number: float
    inverse_overhead_l1: float
    predicted_raw_success: float
    predicted_corrected_success: float
    validation_passed: bool
    component_norms: dict[str, float]
    coherent_compensation_in_forward_circuit: bool
    two_qubit_depth: int

    @classmethod
    def from_channels(
        cls,
        *,
        edge_key: str,
        single_qubit_channels: tuple[Mapping[str, Any], Mapping[str, Any]],
        two_qubit_channel: Mapping[str, Any],
        regularization: float = 1e-5,
        coherent_compensation_in_forward_circuit: bool = False,
        two_qubit_depth: int = 2,
    ) -> GeneratorResponseModel:
        """Construct an inverse without using any benchmark measurement."""

        if regularization < 0.0:
            raise ValueError("regularization must be nonnegative")
        columns = []
        for target in GROVER_TARGETS:
            predicted = simulate_grover_with_generator(
                target,
                single_qubit_channels=single_qubit_channels,
                two_qubit_channel=two_qubit_channel,
                coherent_compensation=coherent_compensation_in_forward_circuit,
                two_qubit_depth=two_qubit_depth,
            )
            columns.append(distribution_vector(predicted))
        response = np.column_stack(columns)
        u_matrix, singular_values, vh_matrix = np.linalg.svd(
            response,
            full_matrices=False,
        )
        factors = singular_values / (
            np.square(singular_values) + float(regularization)
        )
        inverse = (vh_matrix.T * factors) @ u_matrix.T
        condition = float(np.linalg.cond(response))

        corrected_columns = np.column_stack(
            [
                project_probability_simplex(inverse @ response[:, index])
                for index in range(response.shape[1])
            ]
        )
        raw_success = float(np.mean(np.diag(response)))
        corrected_success = float(np.mean(np.diag(corrected_columns)))
        validation_passed = bool(
            np.isfinite(condition)
            and condition < 1.0e6
            and np.all(np.isfinite(inverse))
            and corrected_success + 0.05 >= raw_success
        )

        single_alpha = sum(
            np.linalg.norm(_channel_values(channel, "alpha", ONE_Q_LABELS))
            for channel in single_qubit_channels
        )
        single_gamma = sum(
            np.linalg.norm(_channel_values(channel, "gamma", ONE_Q_LABELS))
            for channel in single_qubit_channels
        )
        kappa = sum(_single_kappa(channel) for channel in single_qubit_channels)
        component_norms = {
            "alpha": float(
                single_alpha
                + np.linalg.norm(
                    _channel_values(two_qubit_channel, "alpha", TWO_Q_LABELS)
                )
            ),
            "gamma": float(
                single_gamma
                + np.linalg.norm(
                    _channel_values(two_qubit_channel, "gamma", TWO_Q_LABELS)
                )
            ),
            "kappa_down": float(kappa),
        }
        return cls(
            edge_key=str(edge_key),
            response_matrix=response,
            inverse_matrix=inverse,
            regularization=float(regularization),
            condition_number=condition,
            inverse_overhead_l1=float(np.linalg.norm(inverse, ord=1)),
            predicted_raw_success=raw_success,
            predicted_corrected_success=corrected_success,
            validation_passed=validation_passed,
            component_norms=component_norms,
            coherent_compensation_in_forward_circuit=bool(
                coherent_compensation_in_forward_circuit
            ),
            two_qubit_depth=int(two_qubit_depth),
        )

    def correct(
        self,
        distribution: Mapping[str, int | float],
        *,
        require_validation: bool = True,
    ) -> dict[str, float]:
        """Apply the learned non-CPTP inverse and project back to probabilities."""

        corrected, _ = self.correct_with_audit(
            distribution,
            require_validation=require_validation,
        )
        return corrected

    def correct_with_audit(
        self,
        distribution: Mapping[str, int | float],
        *,
        require_validation: bool = True,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return the corrected distribution and non-CPTP projection diagnostics."""

        if require_validation and not self.validation_passed:
            raise RuntimeError(
                f"Learned response inverse for {self.edge_key} failed validation"
            )
        quasiprobabilities = self.inverse_matrix @ distribution_vector(distribution)
        corrected = project_probability_simplex(quasiprobabilities)
        output = {
            bitstring: float(corrected[index])
            for index, bitstring in enumerate(GROVER_TARGETS)
        }
        audit = {
            "inverse_raw_normalization": float(quasiprobabilities.sum()),
            "inverse_raw_negativity": float(
                np.abs(quasiprobabilities[quasiprobabilities < 0.0]).sum()
            ),
            "simplex_projection_l1": float(
                np.linalg.norm(corrected - quasiprobabilities, ord=1)
            ),
        }
        return output, audit

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_key": self.edge_key,
            "algorithm": "grover_2q_exhaustive",
            "two_qubit_depth": self.two_qubit_depth,
            "bitstring_order": list(GROVER_TARGETS),
            "response_matrix": self.response_matrix.tolist(),
            "regularized_inverse_matrix": self.inverse_matrix.tolist(),
            "regularization": self.regularization,
            "condition_number": self.condition_number,
            "inverse_overhead_l1": self.inverse_overhead_l1,
            "predicted_raw_success": self.predicted_raw_success,
            "predicted_corrected_success": self.predicted_corrected_success,
            "validation_passed": self.validation_passed,
            "validation_rule": (
                "finite inverse, condition number < 1e6, and no more than "
                "0.05 predicted mean-success loss before empirical validation"
            ),
            "component_norms": self.component_norms,
            "coherent_compensation_in_forward_circuit": (
                self.coherent_compensation_in_forward_circuit
            ),
            "response_source": (
                "reconstructed_generator_forward_model; no benchmark counts used"
            ),
            "semantics": (
                "regularized full-generator non-CPTP error mitigation; "
                "no added correction gates; not fault-tolerant QEC"
            ),
        }


__all__ = [
    "GROVER_TARGETS",
    "GeneratorResponseModel",
    "build_grover_search_circuit",
    "distribution_vector",
    "grover_operation_sequence",
    "hellinger_fidelity",
    "project_probability_simplex",
    "simulate_grover_with_generator",
]
