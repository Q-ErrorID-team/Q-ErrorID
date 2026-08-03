import json
from pathlib import Path

import pandas as pd

from q_error_id.haiqu_app import ExecutionConfig, HaiquErrorPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_compare_mode_executes_trained_qnn_circuits(tmp_path):
    config = ExecutionConfig(
        device="aer_simulator",
        shots=64,
        seed=47,
        model_mode="compare",
        output_root=tmp_path,
        model_root=PROJECT_ROOT / "artifacts" / "models",
        data_root=PROJECT_ROOT / "artifacts" / "datasets",
    )
    report = HaiquErrorPipeline(config).run(include_benchmark=False)

    reconstructed = pd.read_csv(report.artifacts["reconstructed_channels"])
    assert set(reconstructed["model_kind"]) == {"ridge", "qnn"}
    assert reconstructed.query("model_kind == 'qnn'").shape[0] == 7
    assert reconstructed.query("model_kind == 'qnn'")[
        "model_source"
    ].str.contains("measured_qnn").all()

    deployment = pd.read_csv(report.artifacts["model_deployment"])
    assert deployment.shape[0] == 7
    assert deployment["trained_parameters"].all()
    assert deployment["circuit_name"].str.startswith("trained_qnn_").all()
    assert not deployment["circuit_name"].str.contains(
        "angle_encoded_inference"
    ).any()

    audit = json.loads(Path(report.artifacts["execution_audit"]).read_text())
    assert audit["model_mode_requested"] == "compare"
    assert audit["qnn_executed"] is True
    assert audit["qnn_circuit_count"] == 7
