import numpy as np

from q_error_id.core import (
    analyze_identifiability,
    build_channel,
    extract_features,
    one_qubit_parameters,
    one_qubit_protocol,
    representative_parameters,
    two_qubit_protocol,
)
from q_error_id.core.estimation import recover_parameters


def test_one_and_two_qubit_feature_jacobians_have_full_rank():
    one_parameters = representative_parameters("1Q")
    one_report = analyze_identifiability(one_parameters, one_qubit_protocol())
    assert one_report.rank == 7
    assert one_report.parameter_count == 7
    assert len(one_report.minimal_reliable_indices) == 7

    two_parameters = representative_parameters("CX", basis=("ZI", "IZ", "ZX", "ZZ"))
    two_protocol = two_qubit_protocol(target_features=80)
    two_report = analyze_identifiability(two_parameters, two_protocol)
    gate_only_report = analyze_identifiability(
        two_parameters, two_protocol, include_kappa=False
    )
    assert two_protocol.feature_count == 80
    assert two_report.rank == 10
    assert two_report.parameter_count == 10
    assert len(two_report.minimal_reliable_indices) == 10
    assert gate_only_report.rank == 8
    assert gate_only_report.parameter_count == 8


def test_exact_parameter_recovery_with_numerical_optimizer():
    target = one_qubit_parameters(
        alpha=np.array([0.045, -0.03, 0.02]),
        gamma=np.array([0.01, 0.014, 0.007]),
        kappa_down=0.022,
    )
    protocol = one_qubit_protocol()
    target_features = extract_features(build_channel(target), protocol)
    result = recover_parameters(
        one_qubit_parameters(),
        protocol,
        target_features,
    )
    assert result.success
    assert result.residual_norm < 1e-9
    assert np.allclose(result.parameters.as_vector(), target.as_vector(), atol=1e-7)
