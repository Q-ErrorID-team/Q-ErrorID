from types import SimpleNamespace

from q_error_id.haiqu_app.backend import HaiquSession, choose_connected_subgraph
from q_error_id.haiqu_app.config import ExecutionConfig, MitigationMode


def test_connected_subgraph_has_three_tree_edges():
    qubits, edges = choose_connected_subgraph([(0, 1), (1, 2), (2, 3), (5, 6)])
    assert qubits == (0, 1, 2, 3)
    assert len(edges) == 3


def test_no_key_falls_back_without_auth(monkeypatch):
    monkeypatch.delenv("HAIQU_API_KEY", raising=False)
    session = HaiquSession(ExecutionConfig())
    assert session.authenticate() is False
    assert session.cloud_unavailable_reason == "HAIQU_API_KEY is not set"


def test_exact_login_and_experiment_calls(monkeypatch):
    monkeypatch.setenv("HAIQU_API_KEY", "test-key-not-a-real-secret")

    class SDK:
        def __init__(self):
            self.calls = []
            self._experiment = None

        def login(self, **kwargs):
            self.calls.append(("login", kwargs))
            return "Success: Welcome to the Quantum World, test@example.com!"

        def init(self, name):
            self.calls.append(("init", name))
            self._experiment = SimpleNamespace(id="exp-test", name=name)
            return f"Set current experiment to: {name}. View on Dashboard: https://dashboard.haiqu.ai/experiments/exp-test"

    sdk = SDK()
    session = HaiquSession(ExecutionConfig(), sdk=sdk)
    assert session.authenticate() is True
    assert sdk.calls[0] == ("login", {"api_access_key": "test-key-not-a-real-secret"})
    assert sdk.calls[1] == ("init", "Q-ErrorID Hackathon")
    assert session.experiments["root"]["id"] == "exp-test"


def test_mitigation_schema_matches_sdk_131():
    assert MitigationMode.RAW.use_mitigation is False
    readout = MitigationMode.READOUT.error_mitigation_options
    assert readout == {
        "dynamical_decoupling": False,
        "readout_mitigation": True,
        "noise_tailoring": False,
        "advanced_mitigation": False,
    }


def test_ibm_environment_names_map_to_sdk_options(monkeypatch):
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "token-for-test")
    monkeypatch.setenv("IBM_QUANTUM_INSTANCE", "instance-for-test")
    assert ExecutionConfig().ibm_credentials == {
        "ibm_quantum_token": "token-for-test",
        "ibm_quantum_instance": "instance-for-test",
    }


def test_paths_are_rooted_at_output_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("Q_ERROR_ID_MODEL_ROOT", raising=False)
    monkeypatch.delenv("Q_ERROR_ID_DATA_ROOT", raising=False)
    config = ExecutionConfig.from_env(output_root=tmp_path)
    assert config.model_root == tmp_path / "artifacts" / "models"
    assert config.data_root == tmp_path / "artifacts" / "datasets"
