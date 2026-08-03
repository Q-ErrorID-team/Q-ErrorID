import numpy as np

from q_error_id.haiqu_app.circuits import build_one_qubit_diagnostics
from q_error_id.haiqu_app.results import (
    pauli_expectation_from_counts,
    results_to_features,
)


def test_pauli_expectation_respects_qiskit_bit_order():
    counts = {"00": 50, "01": 50}
    assert pauli_expectation_from_counts(counts, "Z", (0,)) == 0.0
    assert pauli_expectation_from_counts(counts, "Z", (1,)) == 1.0
    assert pauli_expectation_from_counts({"00": 50, "11": 50}, "ZZ", (0, 1)) == 1.0


def test_results_to_features_orders_by_feature_index():
    circuits = build_one_qubit_diagnostics(3)
    results = [{"0": 100} for _ in circuits]
    batch = results_to_features(circuits, results, mode="raw")
    assert list(batch.features) == ["q3"]
    assert batch.features["q3"].shape == (18,)
    np.testing.assert_allclose(batch.features["q3"], 1.0)
    assert len(batch.table) == 18
