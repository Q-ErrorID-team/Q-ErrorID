from pathlib import Path


def test_manifest_path_accepts_relative_project_path():
    from scripts.generate_datasets import PROJECT_ROOT, serialize_manifest_path

    assert serialize_manifest_path(Path("artifacts/datasets/example.npz")) == (
        "artifacts/datasets/example.npz"
    )
    assert "\\" not in serialize_manifest_path(
        Path("artifacts/datasets/example.npz")
    )
    assert Path(PROJECT_ROOT).is_absolute()
