import numpy as np

from q_error_id.core.pauli import I, X, Y, Z, pauli_basis, pauli_word


def test_pauli_algebra_and_tensor_words():
    assert np.allclose(X @ X, I)
    assert np.allclose(Y @ Y, I)
    assert np.allclose(Z @ Z, I)
    assert np.allclose(X @ Y, 1.0j * Z)
    assert np.allclose(Y @ Z, 1.0j * X)
    assert np.allclose(Z @ X, 1.0j * Y)
    assert np.allclose(pauli_word("ZX"), np.kron(Z, X))


def test_pauli_basis_is_orthogonal():
    basis = pauli_basis(2)
    gram = np.array([[np.trace(a @ b) for b in basis] for a in basis])
    assert np.allclose(gram, 4.0 * np.eye(16))
