import json

import numpy as np
from qiskit.quantum_info import Statevector

from q_error_id.haiqu_app.models import ModelRepository
from q_error_id.models import (
    HybridQNN,
    LossConfig,
    PhysicalBaseline,
    QNNTrainingConfig,
    export_qiskit_inference,
    load_model_dataset,
    output_spec,
    recommend_strategy,
    vector_to_parameters,
)


def _bounded_targets(rng, samples):
    return np.column_stack(
        [
            rng.uniform(-0.15, 0.15, size=(samples, 3)),
            rng.uniform(0.0, 0.03, size=(samples, 3)),
            rng.uniform(0.0, 0.05, size=(samples, 1)),
        ]
    )


def test_model_dataset_reads_agent1_artifact():
    dataset = load_model_dataset("artifacts/datasets/1q_mixed_channel_train.npz")
    assert dataset.X.shape == (256, 18)
    assert dataset.y.shape == (256, 7)
    assert dataset.spec.names[-1] == "kappa_down"


def test_hybrid_qnn_smoke_fit():
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=(12, 6))
    y = _bounded_targets(rng, 12)
    model = HybridQNN(
        6,
        "1q",
        n_qubits=3,
        n_layers=1,
        seed=11,
    ).fit(
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        config=QNNTrainingConfig(epochs=2, patience=2, seed=13),
        loss_config=LossConfig(),
    )
    prediction = model.predict(x[8:])
    assert prediction.shape == (4, 7)
    assert np.all(prediction[:, :3] <= 0.15)
    assert np.all(prediction[:, 3:6] >= 0.0)
    assert model.history


def test_hybrid_qnn_roundtrip_and_qiskit_circuit(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.uniform(-1.0, 1.0, size=(12, 6))
    y = _bounded_targets(rng, 12)
    model = HybridQNN(
        6,
        "1q",
        n_qubits=3,
        n_layers=1,
        seed=19,
    ).fit(
        x[:8],
        y[:8],
        x[8:],
        y[8:],
        config=QNNTrainingConfig(epochs=2, patience=2, seed=23),
    )
    restored = HybridQNN.load(model.save(tmp_path / "qnn.pt"))
    np.testing.assert_allclose(restored.predict(x[8:]), model.predict(x[8:]))

    circuit = restored.build_qiskit_inference_circuit(x[8], measure=False)
    distribution = Statevector.from_instruction(circuit).probabilities_dict()
    measured_features = restored.quantum_features_from_distribution(distribution)
    standardized = (x[8] - restored.feature_mean) / restored.feature_scale
    exact_features = restored._quantum_features(standardized[None])[0]
    np.testing.assert_allclose(measured_features, exact_features, atol=1e-12)
    np.testing.assert_allclose(
        restored.predict_from_quantum_features(measured_features),
        restored.predict(x[8]),
        atol=1e-12,
    )
    exported = export_qiskit_inference(
        restored,
        tmp_path / "inference",
        reference_features=x[8],
    )
    qasm_text = exported["qasm"].read_text(encoding="utf-8")
    template = json.loads(exported["template"].read_text(encoding="utf-8"))
    assert "measure" in qasm_text
    assert all(name in qasm_text for name in template["parameter_order"])


def test_physical_baselines_roundtrip(tmp_path):
    rng = np.random.default_rng(29)
    x = rng.normal(size=(24, 18))
    y = _bounded_targets(rng, 24)
    for kind in ("ridge", "mlp"):
        model = PhysicalBaseline("1q", kind, hidden_units=3, seed=31).fit(x, y)
        restored = PhysicalBaseline.load(model.save(tmp_path / f"{kind}.pt"))
        np.testing.assert_allclose(
            restored.predict(x[:4]),
            model.predict(x[:4]),
            atol=1e-12,
        )


def test_notebook_parameter_helpers_are_public_and_physical():
    parameters = vector_to_parameters(
        np.array([0.2, 0.0, 0.0, -0.1, 0.01, 0.0, 0.02]),
        output_spec("1q"),
    )
    assert parameters.alpha[0] == 0.15
    assert parameters.gamma[0] == 0.0
    assert recommend_strategy(parameters) == "coherent_inverse_then_mitigation"


def test_saved_ridge_is_consumed_by_haiqu_repository(tmp_path):
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 18))
    y = _bounded_targets(rng, 20)
    model_root = tmp_path / "models"
    artifact = (
        PhysicalBaseline("1q", "ridge").fit(x, y).save(model_root / "ridge_1q.pt")
    )
    assert artifact.exists()
    (model_root / "model_manifest.json").write_text(
        '{"families":{"1q":{"model_paths":{"ridge":"ridge_1q.pt"}}}}',
        encoding="utf-8",
    )
    repository = ModelRepository(model_root, tmp_path / "datasets")
    estimate = repository.predict("1q", x[0])
    assert estimate.trained_model is True
    assert estimate.model_source.endswith("ridge_1q.pt")


def test_saved_qnn_is_consumed_and_measured_by_haiqu_repository(tmp_path):
    rng = np.random.default_rng(37)
    x = rng.normal(size=(14, 6))
    y = _bounded_targets(rng, 14)
    model_root = tmp_path / "models"
    qnn = HybridQNN(6, "1q", n_qubits=3, n_layers=1, seed=41).fit(
        x[:10],
        y[:10],
        x[10:],
        y[10:],
        config=QNNTrainingConfig(epochs=2, patience=2, seed=43),
    )
    artifact = qnn.save(model_root / "qnn_1q.pt")
    (model_root / "model_manifest.json").write_text(
        '{"families":{"1q":{"model_paths":{"qnn_primary":"qnn_1q.pt"}}}}',
        encoding="utf-8",
    )
    repository = ModelRepository(model_root, tmp_path / "datasets")
    circuit = repository.qnn_circuit("1q", x[0], name="trained_qnn_test")
    assert circuit.metadata["trained_parameters"] is True
    distribution = Statevector.from_instruction(circuit.remove_final_measurements(
        inplace=False
    )).probabilities_dict()
    estimate, observables = repository.predict_qnn_distribution(
        "1q",
        distribution,
        execution_source="ideal_statevector_test",
    )
    assert estimate.trained_model is True
    assert "measured_qnn" in estimate.model_source
    assert observables.shape == (2 * qnn.n_qubits,)
    np.testing.assert_allclose(
        [
            estimate.alpha["X"],
            estimate.alpha["Y"],
            estimate.alpha["Z"],
            estimate.gamma["X"],
            estimate.gamma["Y"],
            estimate.gamma["Z"],
            estimate.kappa_down["down"],
        ],
        qnn.predict(x[0]),
        atol=1e-12,
    )
    assert artifact.exists()
