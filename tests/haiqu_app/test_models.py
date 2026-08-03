import numpy as np

from q_error_id.haiqu_app.models import ModelRepository


def test_numpy_model_loading(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    coef = np.zeros((7, 18))
    coef[0, 0] = 2.0
    np.savez(
        model_root / "1q_ridge.npz",
        coef=coef,
        intercept=np.zeros(7),
        feature_mean=np.zeros(18),
        feature_scale=np.ones(18),
    )
    repository = ModelRepository(model_root, tmp_path / "datasets")
    estimate = repository.predict("1q", np.ones(18))
    assert estimate.trained_model is True
    assert estimate.alpha["X"] == 0.15  # clipped to the sampled physical range
    assert estimate.model_source.endswith("1q_ridge.npz")


def test_missing_model_uses_marked_fallback(tmp_path):
    repository = ModelRepository(tmp_path / "models", tmp_path / "datasets")
    estimate = repository.predict("1q", np.array([1, 0, 0] * 6))
    assert estimate.trained_model is False
    assert "fallback" in estimate.model_source
