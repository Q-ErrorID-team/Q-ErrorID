from q_error_id.haiqu_app.atlas import DeviceErrorAtlas
from q_error_id.haiqu_app.models import ChannelEstimate


def estimate(alpha=0.01, gamma=0.02, kappa=0.0):
    return ChannelEstimate(
        alpha={"X": alpha},
        gamma={"X": gamma},
        kappa_down={"down": kappa} if kappa else {},
        model_source="test",
        trained_model=True,
    )


def test_error_atlas_construction_and_json(tmp_path):
    atlas = DeviceErrorAtlas.from_estimates(
        device_id="fake_fez",
        physical_qubits=(0, 1, 2, 3),
        single_qubit_estimates={q: estimate(kappa=0.03) for q in range(4)},
        two_qubit_estimates={
            "q0-q1": estimate(),
            "q1-q2": estimate(),
            "q2-q3": estimate(),
        },
        edge_qubits={
            "q0-q1": (0, 1),
            "q1-q2": (1, 2),
            "q2-q3": (2, 3),
        },
    )
    assert len(atlas.dataframe()) == 7
    path = atlas.save_json(tmp_path / "atlas.json")
    assert '"ground_truth"' not in path.read_text()
    assert atlas.single_qubit_channels[0]["recommended_mitigation"]
