import numpy as np

from q_error_id.core import (
    build_channel,
    channel_to_choi,
    channel_to_kraus,
    one_qubit_parameters,
    two_qubit_parameters,
    validate_channel,
)
from q_error_id.core.representations import kraus_to_superoperator


def _representative_channels():
    yield build_channel(
        one_qubit_parameters(
            alpha=np.array([0.05, -0.04, 0.02]),
            gamma=np.array([0.01, 0.02, 0.005]),
            kappa_down=0.03,
        )
    )
    yield build_channel(
        two_qubit_parameters(
            alpha=np.array([0.04, -0.03, 0.02, 0.01]),
            gamma=np.array([0.01, 0.012, 0.008, 0.015]),
            kappa_down=np.array([0.02, 0.025]),
        )
    )


def test_exponentiated_generators_are_cptp_and_hp():
    for channel in _representative_channels():
        validation = validate_channel(channel)
        assert validation["complete_positive"]
        assert validation["trace_preserving"]
        assert validation["hermiticity_preserving"]
        assert validation["minimum_choi_eigenvalue"] >= -1e-10
        assert validation["trace_preservation_violation"] < 1e-10


def test_normalized_and_unnormalized_choi_traces():
    for channel in _representative_channels():
        normalized = channel_to_choi(channel, normalized=True)
        unnormalized = channel_to_choi(channel, normalized=False)
        assert np.allclose(np.trace(normalized), 1.0)
        assert np.allclose(np.trace(unnormalized), channel.dimension)
        assert np.allclose(unnormalized, channel.dimension * normalized)


def test_kraus_representation_reconstructs_superoperator():
    for channel in _representative_channels():
        kraus = channel_to_kraus(channel)
        completeness = sum(operator.conj().T @ operator for operator in kraus)
        assert np.allclose(completeness, np.eye(channel.dimension), atol=1e-9)
        assert np.allclose(
            kraus_to_superoperator(kraus), channel.superoperator, atol=1e-9
        )
