"""Reproducible synthetic datasets for Q-ErrorID."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .channels import (
    LOCAL_DAMPING_CONVENTION,
    build_channel,
    build_diagnostic_channel,
)
from .parameters import (
    ChannelParameters,
    one_qubit_parameters,
    two_qubit_parameters,
)
from .protocols import (
    PrepareMeasureProtocol,
    ReadoutConfusion,
    extract_features,
    one_qubit_protocol,
    two_qubit_protocol,
)
from .representations import channel_to_choi, channel_to_ptm

SHOT_COUNTS = (8192, 4096, 1024, 256)
FAMILIES = ("1q_mixed_channel", "2q_mixed_channel")


def sample_parameters(
    family: str,
    rng: np.random.Generator,
    *,
    gate_name: str = "CX",
    gate_basis: tuple[str, ...] = ("ZI", "IZ", "ZX", "ZZ"),
) -> ChannelParameters:
    """Draw a physical generator uniformly from the requested project bounds."""

    if family == "1q_mixed_channel":
        return one_qubit_parameters(
            alpha=rng.uniform(-0.15, 0.15, size=3),
            gamma=rng.uniform(0.0, 0.03, size=3),
            kappa_down=rng.uniform(0.0, 0.05),
        )
    if family == "2q_mixed_channel":
        return two_qubit_parameters(
            gate_name=gate_name,
            basis=gate_basis,
            alpha=rng.uniform(-0.15, 0.15, size=4),
            gamma=rng.uniform(0.0, 0.03, size=4),
            kappa_down=rng.uniform(0.0, 0.05, size=2),
        )
    raise ValueError(f"Unknown dataset family: {family!r}")


def _protocol_metadata(protocol: PrepareMeasureProtocol) -> dict[str, Any]:
    return {
        "name": protocol.name,
        "n_qubits": protocol.n_qubits,
        "feature_count": protocol.feature_count,
        "feature_labels": list(protocol.feature_labels),
        "input_labels": [list(labels) for labels in protocol.input_labels],
        "observable_labels": list(protocol.observable_labels),
        "settings": [list(setting) for setting in protocol.settings],
    }


def generate_dataset_split(
    family: str,
    size: int,
    *,
    seed: int,
    protocol: PrepareMeasureProtocol | None = None,
    readout_confusion: ReadoutConfusion | None = None,
    gate_name: str = "CX",
    gate_basis: tuple[str, ...] = ("ZI", "IZ", "ZX", "ZZ"),
) -> dict[str, np.ndarray]:
    """Generate one in-memory split with exact and finite-shot features."""

    if size <= 0:
        raise ValueError("size must be positive")
    rng = np.random.default_rng(seed)
    if protocol is None:
        protocol = (
            one_qubit_protocol()
            if family == "1q_mixed_channel"
            else two_qubit_protocol(
                gate_name=gate_name,
                basis=gate_basis,
                target_features=80,
            )
        )
    if readout_confusion is None:
        readout_confusion = ReadoutConfusion()

    n_qubits = 1 if family == "1q_mixed_channel" else 2
    dimension = 2**n_qubits
    n_alpha = 3 if n_qubits == 1 else 4
    n_gamma = n_alpha
    n_kappa = n_qubits

    arrays: dict[str, np.ndarray] = {
        "X_exact": np.empty((size, protocol.feature_count), dtype=np.float32),
        "X_readout_exact": np.empty((size, protocol.feature_count), dtype=np.float32),
        "y_alpha": np.empty((size, n_alpha), dtype=np.float32),
        "y_gamma": np.empty((size, n_gamma), dtype=np.float32),
        "y_kappa": np.empty((size, n_kappa), dtype=np.float32),
        "channel_ptm": np.empty((size, dimension**2, dimension**2), dtype=np.float32),
        "channel_choi": np.empty(
            (size, dimension**2, dimension**2), dtype=np.complex64
        ),
    }
    for shots in SHOT_COUNTS:
        arrays[f"X_shot_{shots}"] = np.empty(
            (size, protocol.feature_count), dtype=np.float32
        )

    for sample_index in range(size):
        parameters = sample_parameters(
            family,
            rng,
            gate_name=gate_name,
            gate_basis=gate_basis,
        )
        error_channel = build_channel(parameters)
        diagnostic_channel = build_diagnostic_channel(parameters)
        exact = extract_features(diagnostic_channel, protocol)
        readout_exact = extract_features(
            diagnostic_channel,
            protocol,
            readout_confusion=readout_confusion,
        )
        arrays["X_exact"][sample_index] = exact
        arrays["X_readout_exact"][sample_index] = readout_exact
        for shots in SHOT_COUNTS:
            arrays[f"X_shot_{shots}"][sample_index] = extract_features(
                diagnostic_channel,
                protocol,
                shots=shots,
                rng=rng,
                readout_confusion=readout_confusion,
            )
        arrays["y_alpha"][sample_index] = parameters.alpha
        arrays["y_gamma"][sample_index] = parameters.gamma
        if parameters.kappa_down is None:
            arrays["y_kappa"][sample_index] = 0.0
        else:
            arrays["y_kappa"][sample_index] = parameters.kappa_down
        arrays["channel_ptm"][sample_index] = channel_to_ptm(error_channel)
        arrays["channel_choi"][sample_index] = channel_to_choi(
            error_channel, normalized=True
        )

    metadata = {
        "schema_version": "1.0",
        "family": family,
        "size": size,
        "seed": seed,
        "gate_name": "1Q" if n_qubits == 1 else gate_name,
        "gate_basis": (["X", "Y", "Z"] if n_qubits == 1 else list(gate_basis)),
        "duration": 1.0,
        "local_damping_convention": (
            "included_in_joint_generator" if n_qubits == 1 else LOCAL_DAMPING_CONVENTION
        ),
        "choi_normalization": "trace_one",
        "ptm_basis_order": "lexicographic_I_X_Y_Z",
        "vectorization": "column-major",
        "shot_counts": list(SHOT_COUNTS),
        "readout_confusion": readout_confusion.to_dict(),
        "labels_are_pre_readout_physical_parameters": True,
        "feature_channel": (
            "error_channel" if n_qubits == 1 else "implemented_gate=error_after_ideal"
        ),
        "stored_channel_ptm_and_choi": "error_channel_only",
        "parameter_order": ["alpha", "gamma", "kappa_down"],
        "protocol": _protocol_metadata(protocol),
    }
    arrays["metadata"] = np.array(json.dumps(metadata, sort_keys=True), dtype=np.str_)
    return arrays


def save_dataset(path: str | Path, arrays: dict[str, np.ndarray]) -> Path:
    """Serialize one split as a compressed NPZ file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def load_dataset(path: str | Path) -> dict[str, Any]:
    """Load a dataset without pickle and decode its JSON metadata."""

    with np.load(path, allow_pickle=False) as archive:
        result: dict[str, Any] = {key: archive[key] for key in archive.files}
    metadata_scalar = result["metadata"]
    result["metadata"] = json.loads(str(metadata_scalar.item()))
    return result


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 digest for a generated artifact."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
