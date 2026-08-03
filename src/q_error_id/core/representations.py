"""Conversions between Liouville, PTM, Choi, and Kraus representations."""

from __future__ import annotations

import numpy as np

from .channels import QuantumChannel
from .pauli import pauli_basis


def channel_to_liouville(channel: QuantumChannel) -> np.ndarray:
    """Return a defensive copy of the Liouville superoperator."""

    return channel.superoperator.copy()


def channel_to_ptm(channel: QuantumChannel) -> np.ndarray:
    """Return the real Pauli transfer matrix.

    The basis contains unnormalized Pauli words, so
    ``R[i, j] = Tr(P_i E(P_j)) / d`` and the identity channel is exactly the
    identity matrix.
    """

    dimension = channel.dimension
    basis = pauli_basis(channel.n_qubits)
    ptm = np.empty((len(basis), len(basis)), dtype=float)
    outputs = [channel.apply(operator) for operator in basis]
    for row, measured in enumerate(basis):
        for column, output in enumerate(outputs):
            ptm[row, column] = float(
                np.real_if_close(np.trace(measured @ output) / dimension)
            )
    return ptm


def channel_to_choi(channel: QuantumChannel, normalized: bool = True) -> np.ndarray:
    """Return the Choi matrix in reference-system tensor output ordering."""

    dimension = channel.dimension
    choi = np.zeros((dimension * dimension, dimension * dimension), dtype=np.complex128)
    for i in range(dimension):
        for j in range(dimension):
            matrix_unit = np.zeros((dimension, dimension), dtype=np.complex128)
            matrix_unit[i, j] = 1.0
            block = channel.apply(matrix_unit)
            choi[
                i * dimension : (i + 1) * dimension,
                j * dimension : (j + 1) * dimension,
            ] = block
    choi = (choi + choi.conj().T) / 2.0
    return choi / dimension if normalized else choi


def channel_to_choi_state(channel: QuantumChannel) -> np.ndarray:
    """Alias that makes the normalization explicit."""

    return channel_to_choi(channel, normalized=True)


def channel_to_kraus(
    channel: QuantumChannel, tolerance: float = 1e-12
) -> tuple[np.ndarray, ...]:
    """Extract Kraus operators from the Choi matrix for validation only."""

    dimension = channel.dimension
    choi = channel_to_choi(channel, normalized=False)
    eigenvalues, eigenvectors = np.linalg.eigh(choi)
    kraus: list[np.ndarray] = []
    for value, vector in zip(eigenvalues, eigenvectors.T):
        if value > tolerance:
            operator = np.sqrt(value) * vector.reshape((dimension, dimension)).T
            kraus.append(operator)
    return tuple(kraus)


def kraus_to_superoperator(kraus: tuple[np.ndarray, ...]) -> np.ndarray:
    """Construct a Liouville superoperator from Kraus operators."""

    if not kraus:
        raise ValueError("At least one Kraus operator is required")
    return sum(np.kron(operator.conj(), operator) for operator in kraus)


def partial_trace_choi_output(choi: np.ndarray, dimension: int) -> np.ndarray:
    """Trace out the channel-output factor of an unnormalized Choi matrix."""

    tensor = np.asarray(choi).reshape(dimension, dimension, dimension, dimension)
    return np.trace(tensor, axis1=1, axis2=3)


def validate_channel(
    channel: QuantumChannel, tolerance: float = 1e-9
) -> dict[str, float | bool]:
    """Numerically test CP, TP, and Hermiticity preservation."""

    dimension = channel.dimension
    choi = channel_to_choi(channel, normalized=False)
    hermiticity_violation = float(np.linalg.norm(choi - choi.conj().T))
    eigenvalues = np.linalg.eigvalsh((choi + choi.conj().T) / 2.0)
    minimum_eigenvalue = float(eigenvalues.min())
    tp_residual = partial_trace_choi_output(choi, dimension) - np.eye(dimension)
    tp_violation = float(np.linalg.norm(tp_residual))
    return {
        "complete_positive": minimum_eigenvalue >= -tolerance,
        "trace_preserving": tp_violation <= tolerance,
        "hermiticity_preserving": hermiticity_violation <= tolerance,
        "minimum_choi_eigenvalue": minimum_eigenvalue,
        "trace_preservation_violation": tp_violation,
        "hermiticity_violation": hermiticity_violation,
    }
