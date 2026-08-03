import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

from q_error_id.haiqu_app.algorithm_benchmark import (
    GROVER_TARGETS,
    GeneratorResponseModel,
    build_grover_search_circuit,
)
from q_error_id.haiqu_app.atlas import DeviceErrorAtlas
from q_error_id.haiqu_app.backend import HaiquSession
from q_error_id.haiqu_app.config import ExecutionConfig
from q_error_id.haiqu_app.models import ChannelEstimate
from q_error_id.haiqu_app.pipeline import HaiquErrorPipeline


def _single_channel(scale: float) -> dict:
    return {
        "alpha": {
            "X": 0.025 * scale,
            "Y": -0.018 * scale,
            "Z": 0.012 * scale,
        },
        "gamma": {
            "X": 0.006 * scale,
            "Y": 0.004 * scale,
            "Z": 0.008 * scale,
        },
        "kappa_down": {"down": 0.012 * scale},
        "trained_model": True,
    }


def _two_qubit_channel() -> dict:
    return {
        "alpha": {
            "ZI": 0.021,
            "IZ": -0.013,
            "ZX": 0.018,
            "ZZ": 0.011,
        },
        "gamma": {
            "ZI": 0.005,
            "IZ": 0.004,
            "ZX": 0.007,
            "ZZ": 0.006,
        },
        "kappa_down": {},
        "trained_model": True,
    }


def test_all_two_qubit_grover_targets_are_deterministic():
    for depth in (2, 4, 8, 16):
        for target in GROVER_TARGETS:
            circuit = build_grover_search_circuit(
                target,
                width=2,
                two_qubit_depth=depth,
            )
            state = Statevector.from_instruction(
                circuit.remove_final_measurements(inplace=False)
            )
            distribution = state.probabilities_dict()
            assert np.isclose(
                distribution.get(target, 0.0),
                1.0,
                atol=1e-12,
            )
            assert circuit.metadata["logical_two_qubit_gate_count"] == depth
            assert circuit.metadata["requested_two_qubit_depth"] == depth


def test_coherent_inverse_is_inserted_at_gate_locations_not_only_at_end():
    corrected = build_grover_search_circuit(
        "10",
        width=2,
        single_qubit_channels=(_single_channel(1.0), _single_channel(0.8)),
        two_qubit_channel=_two_qubit_channel(),
    )
    operations = [
        instruction.operation.name
        for instruction in corrected.data
        if instruction.operation.name not in {"barrier", "measure"}
    ]
    assert corrected.metadata["coherent_correction"] is True
    assert corrected.metadata["correction_at_end_only"] is False
    assert corrected.metadata["coherent_correction_locations"] > 2
    assert operations[0] == "h"
    assert operations[1] == "unitary"
    assert operations.count("cx") == 2
    assert operations.count("unitary") == corrected.metadata[
        "coherent_correction_locations"
    ]


def test_full_generator_response_inverse_uses_alpha_gamma_and_kappa():
    model = GeneratorResponseModel.from_channels(
        edge_key="q0-q1",
        single_qubit_channels=(_single_channel(1.0), _single_channel(0.7)),
        two_qubit_channel=_two_qubit_channel(),
    )
    assert model.validation_passed
    assert model.predicted_corrected_success > model.predicted_raw_success
    assert model.component_norms["alpha"] > 0.0
    assert model.component_norms["gamma"] > 0.0
    assert model.component_norms["kappa_down"] > 0.0

    corrected_success = []
    for index, target in enumerate(GROVER_TARGETS):
        predicted = {
            bitstring: float(model.response_matrix[row, index])
            for row, bitstring in enumerate(GROVER_TARGETS)
        }
        corrected, audit = model.correct_with_audit(predicted)
        corrected_success.append(corrected[target])
        assert np.isfinite(audit["simplex_projection_l1"])
        assert audit["inverse_raw_negativity"] >= 0.0
    assert np.mean(corrected_success) > 0.99


def test_response_artifact_states_non_qec_semantics():
    model = GeneratorResponseModel.from_channels(
        edge_key="q0-q1",
        single_qubit_channels=(_single_channel(1.0), _single_channel(0.7)),
        two_qubit_channel=_two_qubit_channel(),
    )
    payload = model.to_dict()
    assert payload["response_source"].endswith("no benchmark counts used")
    assert "not fault-tolerant QEC" in payload["semantics"]


def test_pipeline_benchmark_uses_full_inverse_without_gate_overhead(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv("HAIQU_API_KEY", raising=False)
    config = ExecutionConfig(
        device="aer_simulator",
        shots=64,
        seed=31,
        output_root=tmp_path,
    )
    session = HaiquSession(config)
    session.authenticate()
    selected = session.select_device(config.device)
    qubits, edges = session.subgraph()

    singles = {
        qubit: ChannelEstimate(
            alpha=_single_channel(0.5)["alpha"],
            gamma=_single_channel(0.5)["gamma"],
            kappa_down=_single_channel(0.5)["kappa_down"],
            model_source="test",
            trained_model=True,
        )
        for qubit in qubits
    }
    edge_qubits = {f"q{left}-q{right}": (left, right) for left, right in edges}
    doubles = {
        edge_key: ChannelEstimate(
            alpha=_two_qubit_channel()["alpha"],
            gamma=_two_qubit_channel()["gamma"],
            kappa_down={},
            model_source="test",
            trained_model=True,
        )
        for edge_key in edge_qubits
    }
    atlas = DeviceErrorAtlas.from_estimates(
        device_id=selected.id,
        physical_qubits=qubits,
        single_qubit_estimates=singles,
        two_qubit_estimates=doubles,
        edge_qubits=edge_qubits,
    )
    (
        summary,
        details,
        validation,
        seed_summary,
        _,
        response,
        _,
    ) = HaiquErrorPipeline(
        config,
        session=session,
    )._benchmark(atlas, qubits, edges)

    assert summary.shape[0] == 7
    raw = summary.query("scenario == 'raw_haiqu_execution'").iloc[0]
    learned = summary.query(
        "scenario == 'learned_generator_correction_only'"
    ).iloc[0]
    assert learned["two_qubit_gate_count"] == raw["two_qubit_gate_count"]
    assert learned["correction_components"] == "alpha,gamma,kappa_down"
    learned_details = details.query(
        "scenario == 'learned_generator_correction_only'"
    )
    assert learned_details.shape[0] == 48
    assert (learned_details["coherent_correction_locations"] == 0).all()
    assert response["uses_benchmark_counts_for_response_fit"] is False
    assert response["uses_evaluation_counts_for_stack_selection"] is False
    assert response["correction_stack"]["added_physical_correction_gates"] == 0
    assert len(response["validation"]["per_edge_depth"]) == 12
    assert set(response["two_qubit_depths"]) == {2, 4, 8, 16}
    full_details = details.query(
        "scenario == 'learned_readout_plus_generator_correction'"
    )
    assert full_details.shape[0] == 48
    assert full_details["status"].str.startswith("executed").all()
    for response_key, decision in response["validation"][
        "per_edge_depth"
    ].items():
        applied = full_details.loc[
            full_details["response_key"] == response_key,
            "generator_correction_applied",
        ]
        assert applied.eq(decision["generator_enabled"]).all()
    assert validation["split"].eq("validation").all()
    assert {
        "generator_enabled_for_evaluation",
        "selected_stack_for_evaluation",
    }.issubset(validation.columns)
    assert seed_summary["repeat_index"].eq(0).all()
    assert pd.isna(raw["tvd_ci95_low"])
    assert pd.isna(raw["paired_tvd_improvement_ci95_low"])
    assert set(details["requested_two_qubit_depth"]) == {2, 4, 8, 16}
    learned_status = summary.query(
        "scenario == 'learned_readout_plus_generator_correction'"
    )["status"].iloc[0]
    assert learned_status in {
        "executed_validation_gated_learned_correction",
        "executed_readout_only_fallback_no_generator",
    }
