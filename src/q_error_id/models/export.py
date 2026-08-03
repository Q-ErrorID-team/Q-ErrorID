"""Export the trained quantum feature circuit for Qiskit/Haiqu deployment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import Parameter

from .estimators import HybridQNN


def export_qiskit_inference(
    model: HybridQNN,
    path_prefix: str | Path,
    *,
    reference_features: np.ndarray,
) -> dict[str, Path]:
    """Write a parameterized OpenQASM circuit and its feature-binding contract."""

    if not model._fitted:
        raise RuntimeError("The QNN must be trained before export")
    prefix = Path(path_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    angle_count = model.n_layers * model.n_qubits * 2
    data_angles = [Parameter(f"data_{index}") for index in range(angle_count)]
    circuit = QuantumCircuit(
        model.n_qubits,
        model.n_qubits,
        name=f"q_error_id_{model.family}",
    )
    index = 0
    for layer in range(model.n_layers):
        for qubit in range(model.n_qubits):
            circuit.ry(
                data_angles[index] + float(model.theta[layer, qubit, 0]),
                qubit,
            )
            index += 1
            circuit.rz(
                data_angles[index] + float(model.theta[layer, qubit, 1]),
                qubit,
            )
            index += 1
        for qubit in range(model.n_qubits):
            circuit.cx(qubit, (qubit + 1) % model.n_qubits)
    circuit.measure(range(model.n_qubits), range(model.n_qubits))

    qasm_path = prefix.with_suffix(".qasm")
    qasm_path.write_text(qasm3.dumps(circuit), encoding="utf-8")

    reference = np.asarray(reference_features, dtype=float).reshape(1, -1)
    standardized = (reference - model.feature_mean) / model.feature_scale
    encoded = model._encode_angles(standardized).reshape(-1)
    template = {
        "schema_version": "1.0",
        "family": model.family,
        "qasm_file": qasm_path.name,
        "circuit_role": "trained_qnn_inference",
        "measurement_bit_order": "Qiskit little-endian; qubit 0 is rightmost",
        "parameter_order": [str(parameter) for parameter in data_angles],
        "feature_transform": {
            "kind": "standardize_project_tanh",
            "feature_mean": model.feature_mean.tolist(),
            "feature_scale": model.feature_scale.tolist(),
            "projection": model.projection.tolist(),
            "angle_formula": "pi*tanh(projection @ standardized_features)",
        },
        "trained_theta": model.theta.tolist(),
        "reference_binding": {
            str(parameter): float(value)
            for parameter, value in zip(data_angles, encoded)
        },
        "readout": {
            "observables": model.architecture()["measurements"],
            "quantum_feature_mean": model.quantum_feature_mean.tolist(),
            "quantum_feature_scale": model.quantum_feature_scale.tolist(),
            "coefficient": model.coefficients.tolist(),
            "intercept": model.intercept.tolist(),
            "physical_output_bounds": {
                name: [float(low), float(high)]
                for name, low, high in zip(
                    model.spec.names,
                    model.spec.lower_bounds,
                    model.spec.upper_bounds,
                )
            },
        },
    }
    template_path = prefix.with_suffix(".qiskit.json")
    template_path.write_text(
        json.dumps(template, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"qasm": qasm_path, "template": template_path}
