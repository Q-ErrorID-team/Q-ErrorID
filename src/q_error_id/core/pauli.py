"""Pauli operators, product states, and small linear-algebra helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

import numpy as np

I = np.eye(2, dtype=np.complex128)
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}
PAULI_ORDER = ("I", "X", "Y", "Z")


def kron_all(operators: Iterable[np.ndarray]) -> np.ndarray:
    """Return the Kronecker product in the supplied left-to-right order."""

    result = np.array([[1.0]], dtype=np.complex128)
    for operator in operators:
        result = np.kron(result, np.asarray(operator, dtype=np.complex128))
    return result


def pauli_word(label: str) -> np.ndarray:
    """Return the matrix for a Pauli word such as ``"ZX"``."""

    normalized = label.upper()
    if not normalized or any(letter not in PAULI for letter in normalized):
        raise ValueError(f"Invalid Pauli word: {label!r}")
    return kron_all(PAULI[letter] for letter in normalized)


def pauli_labels(n_qubits: int, include_identity: bool = True) -> tuple[str, ...]:
    """Return lexicographically ordered tensor-product Pauli labels."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    labels = tuple("".join(chars) for chars in product(PAULI_ORDER, repeat=n_qubits))
    if include_identity:
        return labels
    identity = "I" * n_qubits
    return tuple(label for label in labels if label != identity)


def pauli_basis(n_qubits: int, include_identity: bool = True) -> tuple[np.ndarray, ...]:
    """Return the unnormalized tensor-product Pauli basis."""

    return tuple(
        pauli_word(label) for label in pauli_labels(n_qubits, include_identity)
    )


def pauli_eigenstate(axis: str, eigenvalue: int) -> np.ndarray:
    """Return a one-qubit Pauli eigenstate as a density matrix."""

    axis = axis.upper()
    if axis not in ("X", "Y", "Z"):
        raise ValueError("axis must be X, Y, or Z")
    if eigenvalue not in (-1, 1):
        raise ValueError("eigenvalue must be +1 or -1")
    return (I + eigenvalue * PAULI[axis]) / 2.0


def state_from_label(label: str) -> np.ndarray:
    """Parse ``X+``, ``X-``, ``Y+``, ``Y-``, ``Z+``, or ``Z-``."""

    normalized = label.strip().upper()
    if len(normalized) != 2 or normalized[0] not in "XYZ" or normalized[1] not in "+-":
        raise ValueError(f"Invalid Pauli eigenstate label: {label!r}")
    return pauli_eigenstate(normalized[0], 1 if normalized[1] == "+" else -1)


def product_state(labels: Sequence[str]) -> np.ndarray:
    """Return a product density matrix from one-qubit state labels."""

    if not labels:
        raise ValueError("At least one state label is required")
    return kron_all(state_from_label(label) for label in labels)


def vec(matrix: np.ndarray) -> np.ndarray:
    """Column-stack a matrix."""

    return np.asarray(matrix, dtype=np.complex128).reshape(-1, order="F")


def unvec(vector: np.ndarray, dimension: int) -> np.ndarray:
    """Undo :func:`vec` for a square matrix."""

    return np.asarray(vector, dtype=np.complex128).reshape(
        (dimension, dimension), order="F"
    )
