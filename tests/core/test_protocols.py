import numpy as np

from q_error_id.core import (
    ReadoutConfusion,
    build_channel,
    expectation_from_choi,
    extract_features,
    one_qubit_parameters,
    one_qubit_protocol,
)
from q_error_id.core.protocols import sample_expectations


def test_choi_and_prepare_measure_expectations_agree():
    channel = build_channel(
        one_qubit_parameters(
            alpha=np.array([0.04, -0.025, 0.015]),
            gamma=np.array([0.01, 0.012, 0.006]),
            kappa_down=0.02,
        )
    )
    protocol = one_qubit_protocol()
    direct = extract_features(channel, protocol)
    from_choi = np.array(
        [
            expectation_from_choi(
                channel,
                protocol.input_states[input_index],
                protocol.observables[observable_index],
            )
            for input_index, observable_index in protocol.settings
        ]
    )
    assert np.allclose(direct, from_choi, atol=1e-11)


def test_finite_shot_estimates_converge():
    expectations = np.array([-0.8, -0.25, 0.0, 0.35, 0.9])
    rng = np.random.default_rng(20260724)
    errors_256 = []
    errors_8192 = []
    for _ in range(300):
        errors_256.append(
            np.mean((sample_expectations(expectations, 256, rng) - expectations) ** 2)
        )
        errors_8192.append(
            np.mean((sample_expectations(expectations, 8192, rng) - expectations) ** 2)
        )
    assert np.mean(errors_8192) < np.mean(errors_256) / 20.0


def test_asymmetric_readout_confusion_keeps_physical_labels_unchanged():
    channel = build_channel(one_qubit_parameters())
    protocol = one_qubit_protocol()
    confusion = ReadoutConfusion(
        p_plus_to_minus=0.1,
        p_minus_to_plus=0.03,
    )
    physical = extract_features(channel, protocol)
    corrupted = extract_features(channel, protocol, readout_confusion=confusion)
    z_plus_z_index = protocol.feature_labels.index("Z+->Z")
    assert np.isclose(physical[z_plus_z_index], 1.0)
    assert np.isclose(corrupted[z_plus_z_index], 0.8)
