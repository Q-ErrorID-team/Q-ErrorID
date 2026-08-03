from q_error_id.core.datasets import (
    generate_dataset_split,
    load_dataset,
    save_dataset,
)
from q_error_id.core.protocols import ReadoutConfusion


def test_dataset_serialization_and_loading(tmp_path):
    arrays = generate_dataset_split(
        "1q_mixed_channel",
        3,
        seed=1234,
        readout_confusion=ReadoutConfusion(0.01, 0.02),
    )
    path = save_dataset(tmp_path / "tiny.npz", arrays)
    loaded = load_dataset(path)
    required = {
        "X_exact",
        "X_shot_8192",
        "X_shot_4096",
        "X_shot_1024",
        "X_shot_256",
        "y_alpha",
        "y_gamma",
        "y_kappa",
        "channel_ptm",
        "channel_choi",
        "metadata",
    }
    assert required.issubset(loaded)
    assert loaded["X_exact"].shape == (3, 18)
    assert loaded["channel_ptm"].shape == (3, 4, 4)
    assert loaded["channel_choi"].shape == (3, 4, 4)
    assert loaded["metadata"]["labels_are_pre_readout_physical_parameters"]
    assert loaded["metadata"]["readout_confusion"]["p_plus_to_minus"] == 0.01
