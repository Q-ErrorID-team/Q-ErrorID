"""Lindblad generators and finite-time quantum channels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm

from .parameters import ChannelParameters
from .pauli import I, pauli_word, unvec, vec

LOCAL_DAMPING_CONVENTION = "local_before_gate_error"
SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
IDEAL_CX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.complex128,
)
IDEAL_CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)


def _commutator_superoperator(hamiltonian: np.ndarray) -> np.ndarray:
    dimension = hamiltonian.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    return -1.0j * (np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity))


def _lindblad_superoperator(collapse: np.ndarray) -> np.ndarray:
    """Return ``D[L]`` under column-major vectorization."""

    dimension = collapse.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    product = collapse.conj().T @ collapse
    return (
        np.kron(collapse.conj(), collapse)
        - 0.5 * np.kron(identity, product)
        - 0.5 * np.kron(product.T, identity)
    )


def _pauli_dissipator(pauli: np.ndarray) -> np.ndarray:
    dimension = pauli.shape[0]
    return np.kron(pauli.T, pauli) - np.eye(dimension * dimension, dtype=np.complex128)


def generator_liouvillian(parameters: ChannelParameters) -> np.ndarray:
    """Construct the requested one- or two-qubit GKSL generator."""

    dimension = 2**parameters.n_qubits
    hamiltonian = np.zeros((dimension, dimension), dtype=np.complex128)
    for coefficient, label in zip(parameters.alpha, parameters.coherent_labels):
        hamiltonian += 0.5 * coefficient * pauli_word(label)

    generator = _commutator_superoperator(hamiltonian)
    for rate, label in zip(parameters.gamma, parameters.stochastic_labels):
        generator += rate * _pauli_dissipator(pauli_word(label))

    if parameters.n_qubits == 1 and parameters.kappa_down is not None:
        generator += parameters.kappa_down[0] * _lindblad_superoperator(SIGMA_MINUS)
    return generator


def local_damping_liouvillian(rates: np.ndarray) -> np.ndarray:
    """Construct independent amplitude damping on two qubits."""

    values = np.asarray(rates, dtype=float).reshape(-1)
    if values.size != 2 or np.any(values < 0.0):
        raise ValueError("Two nonnegative local damping rates are required")
    lowering_0 = np.kron(SIGMA_MINUS, I)
    lowering_1 = np.kron(I, SIGMA_MINUS)
    return values[0] * _lindblad_superoperator(lowering_0) + values[
        1
    ] * _lindblad_superoperator(lowering_1)


@dataclass(frozen=True)
class QuantumChannel:
    """A finite-dimensional channel stored as a Liouville superoperator."""

    superoperator: np.ndarray
    n_qubits: int
    metadata: dict[str, Any]
    parameters: ChannelParameters | None = None

    def __post_init__(self) -> None:
        dimension = 2**self.n_qubits
        array = np.asarray(self.superoperator, dtype=np.complex128)
        if array.shape != (dimension * dimension, dimension * dimension):
            raise ValueError("Superoperator has an incompatible shape")
        object.__setattr__(self, "superoperator", array)

    @property
    def dimension(self) -> int:
        return 2**self.n_qubits

    def apply(self, density_matrix: np.ndarray) -> np.ndarray:
        """Apply the channel to a density matrix."""

        rho = np.asarray(density_matrix, dtype=np.complex128)
        if rho.shape != (self.dimension, self.dimension):
            raise ValueError("Density matrix has an incompatible shape")
        output = unvec(self.superoperator @ vec(rho), self.dimension)
        return (
            (output + output.conj().T) / 2.0
            if np.allclose(rho, rho.conj().T)
            else output
        )

    def then(self, later: QuantumChannel) -> QuantumChannel:
        """Compose channels so that ``later`` acts after ``self``."""

        if self.n_qubits != later.n_qubits:
            raise ValueError("Cannot compose channels with different dimensions")
        return QuantumChannel(
            superoperator=later.superoperator @ self.superoperator,
            n_qubits=self.n_qubits,
            metadata={
                "composition": [self.metadata, later.metadata],
                "vectorization": "column-major",
            },
        )


def build_channel(
    parameters: ChannelParameters,
    *,
    duration: float = 1.0,
    local_damping_convention: str = LOCAL_DAMPING_CONVENTION,
) -> QuantumChannel:
    """Exponentiate a physical generator into a CPTP channel.

    The two-qubit convention is ``local_before_gate_error``: independent local
    damping acts first, followed by the gate-specific error channel.  The
    alternative ``local_after_gate_error`` is supported for controlled studies.
    """

    if duration < 0.0:
        raise ValueError("duration must be nonnegative")
    gate_generator = generator_liouvillian(parameters)
    gate_map = expm(duration * gate_generator)
    metadata: dict[str, Any] = {
        "gate_name": parameters.gate_name,
        "qubits": list(parameters.qubits),
        "duration": float(duration),
        "vectorization": "column-major",
        "represents": "error_channel",
        "coherent_labels": list(parameters.coherent_labels),
        "stochastic_labels": list(parameters.stochastic_labels),
    }

    if parameters.n_qubits == 1:
        metadata["local_damping_convention"] = "included_in_joint_generator"
        return QuantumChannel(gate_map, 1, metadata, parameters)

    local_rates = (
        np.zeros(2)
        if parameters.kappa_down is None
        else np.asarray(parameters.kappa_down, dtype=float)
    )
    local_map = expm(duration * local_damping_liouvillian(local_rates))
    if local_damping_convention == "local_before_gate_error":
        total_map = gate_map @ local_map
    elif local_damping_convention == "local_after_gate_error":
        total_map = local_map @ gate_map
    else:
        raise ValueError(
            "local_damping_convention must be local_before_gate_error "
            "or local_after_gate_error"
        )
    metadata["local_damping_convention"] = local_damping_convention
    metadata["local_damping_rates"] = local_rates.tolist()
    return QuantumChannel(total_map, 2, metadata, parameters)


def identity_channel(n_qubits: int) -> QuantumChannel:
    """Return an exact identity channel."""

    dimension = 2**n_qubits
    return QuantumChannel(
        np.eye(dimension * dimension, dtype=np.complex128),
        n_qubits,
        {"gate_name": "identity", "vectorization": "column-major"},
    )


def unitary_channel(unitary: np.ndarray, *, gate_name: str) -> QuantumChannel:
    """Return the channel ``rho -> U rho U†`` for a validated unitary."""

    matrix = np.asarray(unitary, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix")
    dimension = matrix.shape[0]
    n_qubits = int(np.log2(dimension))
    if 2**n_qubits != dimension:
        raise ValueError("unitary dimension must be a power of two")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(dimension), atol=1e-10):
        raise ValueError("matrix is not unitary")
    return QuantumChannel(
        np.kron(matrix.conj(), matrix),
        n_qubits,
        {
            "gate_name": gate_name.upper(),
            "represents": "ideal_gate",
            "vectorization": "column-major",
        },
    )


def ideal_gate_channel(gate_name: str) -> QuantumChannel:
    """Return the ideal gate used by a diagnostic circuit."""

    normalized = gate_name.upper()
    if normalized == "CX":
        return unitary_channel(IDEAL_CX, gate_name=normalized)
    if normalized == "CZ":
        return unitary_channel(IDEAL_CZ, gate_name=normalized)
    raise ValueError(f"No ideal diagnostic gate is defined for {gate_name!r}")


def build_diagnostic_channel(
    parameters: ChannelParameters,
    *,
    duration: float = 1.0,
    local_damping_convention: str = LOCAL_DAMPING_CONVENTION,
) -> QuantumChannel:
    """Compose the target error after the ideal gate seen by diagnostics.

    The learned parameters still describe only the error channel.  This helper
    supplies the implemented map ``E_error o U_ideal`` whose prepare-and-measure
    expectations match a circuit containing the physical gate.
    """

    error = build_channel(
        parameters,
        duration=duration,
        local_damping_convention=local_damping_convention,
    )
    if parameters.n_qubits == 1:
        return error
    ideal = ideal_gate_channel(parameters.gate_name)
    return QuantumChannel(
        superoperator=error.superoperator @ ideal.superoperator,
        n_qubits=2,
        metadata={
            "gate_name": parameters.gate_name,
            "represents": "implemented_noisy_gate",
            "composition": "error_after_ideal",
            "error_metadata": error.metadata,
            "ideal_metadata": ideal.metadata,
            "vectorization": "column-major",
        },
        parameters=parameters,
    )
