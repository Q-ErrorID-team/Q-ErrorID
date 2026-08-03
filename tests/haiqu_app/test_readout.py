import numpy as np

from q_error_id.haiqu_app.circuits import (
    build_readout_calibration_circuits,
)
from q_error_id.haiqu_app.readout import (
    ReadoutAssignment,
    ReadoutCalibrationBundle,
    marginalize_distribution,
)


def test_readout_calibration_bank_has_twenty_independent_circuits():
    qubits = (0, 1, 2, 3)
    edges = ((0, 1), (1, 2), (2, 3))
    circuits = build_readout_calibration_circuits(qubits, edges)

    assert len(circuits) == 20
    assert sum(
        len(circuit.metadata["physical_qubits"]) == 1 for circuit in circuits
    ) == 8
    assert sum(
        len(circuit.metadata["physical_qubits"]) == 2 for circuit in circuits
    ) == 12
    assert all(
        circuit.metadata["calibration_type"] == "readout_assignment"
        for circuit in circuits
    )


def test_regularized_assignment_inverse_recovers_known_distribution():
    matrix = np.asarray(
        [
            [0.96, 0.04],
            [0.04, 0.96],
        ]
    )
    assignment = ReadoutAssignment.from_matrix(
        key="q0",
        physical_qubits=(0,),
        matrix=matrix,
        regularization=1e-10,
    )
    true = np.asarray([0.31, 0.69])
    measured = matrix @ true
    corrected, audit = assignment.correct_with_audit(
        {"0": measured[0], "1": measured[1]}
    )

    assert np.allclose([corrected["0"], corrected["1"]], true, atol=1e-8)
    assert assignment.validation_passed
    assert audit["readout_raw_negativity"] == 0.0


def test_bundle_reconstructs_edge_matrix_and_corrects_qiskit_bit_order():
    qubits = (0, 1, 2, 3)
    edges = ((0, 1), (1, 2), (2, 3))
    circuits = build_readout_calibration_circuits(qubits, edges)
    results = []
    for circuit in circuits:
        state = circuit.metadata["prepared_state"]
        active = circuit.metadata["measurement_clbits"]
        full = ["0"] * 4
        for logical_index, clbit in enumerate(active):
            full[3 - clbit] = state[-1 - logical_index]
        results.append({"".join(full): 1024})

    bundle = ReadoutCalibrationBundle.from_results(circuits, results)
    corrected, _ = bundle.correct_edge("q1-q2", {"10": 700, "01": 300})

    assert bundle.validation_passed
    assert np.isclose(corrected["10"], 0.7)
    assert np.isclose(corrected["01"], 0.3)
    assert marginalize_distribution({"0101": 10}, (0, 2)) == {
        "00": 0.0,
        "01": 0.0,
        "10": 0.0,
        "11": 1.0,
    }


def test_probability_payload_uses_explicit_calibration_shot_count():
    qubits = (0, 1, 2, 3)
    edges = ((0, 1), (1, 2), (2, 3))
    circuits = build_readout_calibration_circuits(qubits, edges)
    results = []
    for circuit in circuits:
        state = circuit.metadata["prepared_state"]
        active = circuit.metadata["measurement_clbits"]
        full = ["0"] * 4
        for logical_index, clbit in enumerate(active):
            full[3 - clbit] = state[-1 - logical_index]
        results.append({"".join(full): 1.0})

    bundle = ReadoutCalibrationBundle.from_results(
        circuits,
        results,
        expected_shots=64,
    )

    assert bundle.calibration_shots == 64
