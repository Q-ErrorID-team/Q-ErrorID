import numpy as np
from qiskit.quantum_info import Statevector

from q_error_id.core import (
    build_diagnostic_channel,
    extract_features,
    two_qubit_parameters,
    two_qubit_protocol,
)
from q_error_id.haiqu_app.circuits import (
    REQUIRED_METADATA,
    apply_coherent_correction,
    build_benchmark_circuit,
    build_one_qubit_diagnostics,
    build_two_qubit_diagnostics,
    validate_diagnostic_metadata,
)
from q_error_id.haiqu_app.results import pauli_expectation_from_counts


def test_diagnostic_metadata_contract():
    circuits = build_one_qubit_diagnostics(7)
    circuits += build_two_qubit_diagnostics((7, 8))
    for circuit in circuits:
        validate_diagnostic_metadata(circuit)
        assert REQUIRED_METADATA.issubset(circuit.metadata)


def test_correction_circuit_generation():
    benchmark = build_benchmark_circuit()
    corrected = apply_coherent_correction(
        benchmark,
        {
            0: {
                "alpha": {"X": 0.01, "Y": -0.02, "Z": 0.03},
            }
        },
        {
            "q0-q1": {
                "physical_qubits": [0, 1],
                "alpha": {"ZI": 0.01, "IZ": 0.02, "ZX": 0.03, "ZZ": 0.04},
            }
        },
    )
    assert corrected.metadata["coherent_correction"] is True
    names = corrected.count_ops()
    assert names["rx"] == 1
    assert names["ry"] == 1
    assert names["rzz"] >= 2
    assert names["measure"] == 4


def test_ideal_cx_circuits_match_agent1_diagnostic_contract():
    protocol = two_qubit_protocol(target_features=80)
    parameters = two_qubit_parameters(
        alpha=np.zeros(4),
        gamma=np.zeros(4),
        kappa_down=np.zeros(2),
    )
    expected = extract_features(
        build_diagnostic_channel(parameters),
        protocol,
    )
    circuits = build_two_qubit_diagnostics((0, 1))
    observed = []
    for circuit in circuits:
        unitary_part = circuit.remove_final_measurements(inplace=False)
        state = Statevector.from_instruction(unitary_part)
        observed.append(
            pauli_expectation_from_counts(
                state.probabilities_dict(),
                circuit.metadata["measurement_basis"],
                circuit.metadata["circuit_qubits"],
            )
        )
    assert np.allclose(observed, expected, atol=1e-12)
