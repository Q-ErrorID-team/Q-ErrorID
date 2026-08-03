import numpy as np

from q_error_id.core import (
    build_channel,
    channel_to_choi,
    choi_state_fidelity,
    error_strengths,
    minimum_choi_eigenvalue,
    one_qubit_parameters,
    parameter_mae,
    parameter_rmse,
    relative_ptm_frobenius_error,
    trace_preservation_violation,
    unitarity,
)


def test_parameter_and_channel_metrics():
    target_values = np.array([0.0, 1.0, 2.0])
    predicted_values = np.array([0.0, 2.0, 1.0])
    assert np.isclose(parameter_mae(target_values, predicted_values), 2.0 / 3.0)
    assert np.isclose(
        parameter_rmse(target_values, predicted_values), np.sqrt(2.0 / 3.0)
    )

    identity = build_channel(one_qubit_parameters())
    noisy_parameters = one_qubit_parameters(
        alpha=np.array([0.03, 0.0, 0.0]),
        gamma=np.array([0.01, 0.0, 0.0]),
        kappa_down=0.02,
    )
    noisy = build_channel(noisy_parameters)
    assert np.isclose(choi_state_fidelity(identity, identity), 1.0)
    assert relative_ptm_frobenius_error(identity, noisy) > 0.0
    assert trace_preservation_violation(noisy) < 1e-10
    assert minimum_choi_eigenvalue(noisy) >= -1e-10
    assert np.isclose(unitarity(identity), 1.0)
    assert unitarity(noisy) < 1.0
    assert np.isclose(np.trace(channel_to_choi(noisy)), 1.0)
    strengths = error_strengths(noisy_parameters)
    assert np.isclose(strengths["coherent_l2"], 0.03)
    assert np.isclose(strengths["incoherent_l1"], 0.03)
