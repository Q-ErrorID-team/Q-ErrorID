import numpy as np
from scipy.linalg import expm

from q_error_id.core import (
    build_channel,
    channel_to_ptm,
    one_qubit_parameters,
    two_qubit_parameters,
)
from q_error_id.core.channels import (
    generator_liouvillian,
    local_damping_liouvillian,
)


def test_zero_parameter_identity_channels():
    one = build_channel(one_qubit_parameters())
    two = build_channel(two_qubit_parameters())
    assert np.allclose(one.superoperator, np.eye(4), atol=1e-13)
    assert np.allclose(two.superoperator, np.eye(16), atol=1e-13)
    assert np.allclose(channel_to_ptm(one), np.eye(4), atol=1e-13)
    assert np.allclose(channel_to_ptm(two), np.eye(16), atol=1e-13)


def test_two_qubit_local_damping_convention_is_applied_before_gate_error():
    parameters = two_qubit_parameters(
        alpha=np.array([0.04, -0.02, 0.03, 0.01]),
        gamma=np.array([0.01, 0.005, 0.012, 0.008]),
        kappa_down=np.array([0.02, 0.03]),
    )
    channel = build_channel(parameters)
    gate_map = expm(generator_liouvillian(parameters))
    local_map = expm(local_damping_liouvillian(parameters.kappa_down))
    assert channel.metadata["local_damping_convention"] == "local_before_gate_error"
    assert np.allclose(channel.superoperator, gate_map @ local_map)
    assert not np.allclose(channel.superoperator, local_map @ gate_map)


def test_custom_cz_basis_is_supported():
    basis = ("XI", "IX", "ZZ", "YY")
    parameters = two_qubit_parameters(gate_name="CZ", basis=basis)
    channel = build_channel(parameters)
    assert parameters.coherent_labels == basis
    assert channel.metadata["coherent_labels"] == list(basis)
