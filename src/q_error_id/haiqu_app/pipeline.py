"""End-to-end Haiqu-native Q-ErrorID orchestration."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.stats import t as student_t

from .algorithm_benchmark import (
    GROVER_TARGETS,
    GeneratorResponseModel,
    build_grover_search_circuit,
    hellinger_fidelity,
)
from .atlas import DeviceErrorAtlas
from .backend import HaiquSession
from .circuits import (
    build_one_qubit_diagnostics,
    build_readout_calibration_circuits,
    build_two_qubit_diagnostics,
    diagnostic_table,
    local_circuit_analytics,
)
from .config import ExecutionConfig, MitigationMode
from .models import (
    ONE_Q_LABELS,
    TWO_Q_LABELS,
    ChannelEstimate,
    ModelRepository,
    predict_features_with_agent1,
)
from .readout import ReadoutCalibrationBundle
from .results import (
    FeatureBatch,
    normalize_distribution,
    results_to_features,
    success_probability,
    total_variation_distance,
)


@dataclass
class PipelineReport:
    execution_mode: str
    device_id: str
    physical_qubits: tuple[int, ...]
    experiments: dict[str, dict[str, Any]]
    artifacts: dict[str, str]
    correction_improvement_tvd: float | None
    unsupported_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "device_id": self.device_id,
            "physical_qubits": list(self.physical_qubits),
            "experiments": self.experiments,
            "artifacts": self.artifacts,
            "correction_improvement_tvd": self.correction_improvement_tvd,
            "unsupported_features": self.unsupported_features,
        }


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        default=lambda item: (
            item.tolist() if isinstance(item, np.ndarray) else str(item)
        ),
    )


def _save_dataframe(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _ideal_counts(
    circuits: Sequence[QuantumCircuit], *, shots: int, seed: int
) -> list[dict[str, int]]:
    result = (
        AerSimulator(seed_simulator=seed)
        .run(
            list(circuits),
            shots=shots,
            seed_simulator=seed,
        )
        .result()
    )
    counts = result.get_counts()
    return [dict(counts)] if isinstance(counts, Mapping) else [dict(x) for x in counts]


def _job_metadata(job: Any | None) -> dict[str, Any]:
    if job is None:
        return {"job_id": None, "qpu_cost": None}
    info = getattr(job, "info", None) or {}
    return {
        "job_id": getattr(job, "id", None),
        "qpu_cost": info.get("qpu_cost"),
        "uncertainty": info.get("uncertainty"),
    }


def _mean_ci95(values: Sequence[float]) -> tuple[float, float | None, float | None]:
    """Return a mean and Student-t confidence interval over independent repeats."""

    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), None, None
    mean = float(array.mean())
    if array.size < 2:
        return mean, None, None
    standard_error = float(array.std(ddof=1) / np.sqrt(array.size))
    half_width = float(
        student_t.ppf(0.975, df=array.size - 1) * standard_error
    )
    return mean, mean - half_width, mean + half_width


def _split_repeated_results(
    distributions: Sequence[Mapping[str, int | float]],
    *,
    circuit_count: int,
    repeats: int,
) -> list[list[dict[str, int | float]]]:
    expected = circuit_count * repeats
    if len(distributions) != expected:
        raise RuntimeError(
            f"Received {len(distributions)} distributions; expected {expected}"
        )
    return [
        [
            dict(item)
            for item in distributions[
                repeat * circuit_count : (repeat + 1) * circuit_count
            ]
        ]
        for repeat in range(repeats)
    ]


_PRESENTATION_SCENARIOS = (
    "raw_haiqu_execution",
    "calibrated_readout_mitigation_only",
    "learned_readout_plus_generator_correction",
    "learned_full_correction_plus_haiqu_mitigation",
)


def _depth_sweep_summary(details: pd.DataFrame) -> pd.DataFrame:
    """Summarize fixed-suite repeat means separately at each 2Q depth."""

    rows: list[dict[str, Any]] = []
    executed = details[
        details["status"].fillna("").str.startswith("executed")
        & details["scenario"].isin(_PRESENTATION_SCENARIOS)
    ].copy()
    if executed.empty:
        return pd.DataFrame()
    for (scenario, depth), group in executed.groupby(
        ["scenario", "requested_two_qubit_depth"],
        sort=False,
    ):
        repeat_metrics = (
            group.groupby("repeat_index", as_index=False)
            .agg(
                tvd_to_ideal=("tvd_to_ideal", "mean"),
                success_probability=("success_probability", "mean"),
            )
            .sort_values("repeat_index")
        )
        tvd_mean, tvd_low, tvd_high = _mean_ci95(
            repeat_metrics["tvd_to_ideal"]
        )
        success_mean, success_low, success_high = _mean_ci95(
            repeat_metrics["success_probability"]
        )
        rows.append(
            {
                "scenario": scenario,
                "requested_two_qubit_depth": int(depth),
                "repeat_count": len(repeat_metrics),
                "problem_count_per_repeat": int(
                    group[["edge_key", "target_state"]]
                    .drop_duplicates()
                    .shape[0]
                ),
                "tvd_to_ideal": tvd_mean,
                "tvd_ci95_low": (
                    max(0.0, tvd_low) if tvd_low is not None else None
                ),
                "tvd_ci95_high": (
                    min(1.0, tvd_high) if tvd_high is not None else None
                ),
                "success_probability": success_mean,
                "success_probability_ci95_low": (
                    max(0.0, success_low)
                    if success_low is not None
                    else None
                ),
                "success_probability_ci95_high": (
                    min(1.0, success_high)
                    if success_high is not None
                    else None
                ),
                "ci_method": (
                    "student_t_over_independent_repeats"
                    if len(repeat_metrics) >= 2
                    else "not_available_fewer_than_two_repeats"
                ),
                "depth_sweep_method": "cx_cx_identity_folding",
                "learned_generator_applied_fraction": float(
                    group["generator_correction_applied"]
                    .fillna(False)
                    .astype(bool)
                    .mean()
                ),
            }
        )
    return pd.DataFrame(rows)


class HaiquErrorPipeline:
    """Integrate diagnostic circuits, Agent 2 inference, atlas, and benchmark."""

    def __init__(
        self,
        config: ExecutionConfig,
        *,
        session: HaiquSession | None = None,
    ):
        self.config = config
        self.session = session or HaiquSession(config)
        self.models = ModelRepository(
            config.model_root,
            config.data_root,
            shots=config.shots,
        )
        self.root = Path(config.output_root)
        self.artifact_dir = self.root / "artifacts/haiqu"
        self.result_dir = self.root / "results/haiqu"
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def _progress(self, message: str) -> None:
        if self.config.verbose:
            print(f"[Q-ErrorID] {message}", file=sys.stderr, flush=True)

    def _build_diagnostics(
        self,
        qubits: tuple[int, ...],
        edges: tuple[tuple[int, int], ...],
    ) -> tuple[list[QuantumCircuit], dict[str, tuple[int, int]]]:
        index = {physical: logical for logical, physical in enumerate(qubits)}
        circuits: list[QuantumCircuit] = []
        for physical in qubits:
            circuits.extend(
                build_one_qubit_diagnostics(
                    physical,
                    gate_name="id",
                    width=len(qubits),
                    circuit_qubit=index[physical],
                )
            )
        edge_qubits = {}
        for physical_edge in edges:
            key = f"q{physical_edge[0]}-q{physical_edge[1]}"
            edge_qubits[key] = physical_edge
            circuits.extend(
                build_two_qubit_diagnostics(
                    physical_edge,
                    gate_name="cx",
                    width=len(qubits),
                    circuit_qubits=(
                        index[physical_edge[0]],
                        index[physical_edge[1]],
                    ),
                )
            )
        return circuits, edge_qubits

    def _estimate_batch(
        self,
        batch: FeatureBatch,
        ideal_batch: FeatureBatch,
        *,
        model_kind: str,
        inference_backend: str,
        feature_regime: str = "raw_shot",
        feature_stage: str = "raw",
    ) -> tuple[dict[str, ChannelEstimate], list[dict[str, Any]]]:
        estimates = {}
        rows = []
        for channel_key, features in batch.features.items():
            family = "2q" if "-" in channel_key else "1q"
            ideal = ideal_batch.features.get(channel_key)
            estimate = self.models.predict(
                family,
                features,
                model_kind=model_kind,
                ideal_features=ideal,
                feature_regime=feature_regime,
            )
            prediction = predict_features_with_agent1(estimate, family)
            held_out_rmse = None
            if prediction is not None and prediction.shape == features.shape:
                held_out_rmse = float(np.sqrt(np.mean((prediction - features) ** 2)))
            estimates[channel_key] = estimate
            rows.append(
                {
                    "channel_key": channel_key,
                    "features": _json(features),
                    "predicted_alpha": _json(estimate.alpha),
                    "predicted_gamma": _json(estimate.gamma),
                    "predicted_kappa_down": _json(estimate.kappa_down),
                    "held_out_prediction_rmse": held_out_rmse,
                    "model_kind": model_kind,
                    "feature_regime": feature_regime,
                    "feature_stage": feature_stage,
                    "inference_backend": inference_backend,
                    "model_source": estimate.model_source,
                    "trained_model": estimate.trained_model,
                }
            )
        return estimates, rows

    def _execute_qnn_batch(
        self,
        batch: FeatureBatch,
        qubits: tuple[int, ...],
        edges: tuple[tuple[int, int], ...],
        readout_calibration: ReadoutCalibrationBundle | None = None,
    ) -> tuple[
        dict[str, ChannelEstimate],
        list[dict[str, Any]],
        pd.DataFrame,
    ]:
        """Execute restored trained QNN circuits and decode their readouts."""

        contexts: list[tuple[str, str, np.ndarray, QuantumCircuit]] = []
        for channel_key, features in batch.features.items():
            family = "2q" if "-" in channel_key else "1q"
            model = self.models.qnn_model(family, required=True)
            if model.n_qubits > len(qubits):
                raise RuntimeError(
                    f"The trained {family} QNN needs {model.n_qubits} qubits, "
                    f"but the selected demonstration subgraph has {len(qubits)}"
                )
            circuit = self.models.qnn_circuit(
                family,
                features,
                name=f"trained_qnn_{family}_{channel_key.replace('-', '_')}",
                physical_qubits=qubits[: model.n_qubits],
            )
            contexts.append((channel_key, family, features, circuit))

        circuits = [context[3] for context in contexts]
        if self.session.cloud_enabled:
            logged = self.session.log_circuits(circuits, group="model")
            execution_circuits, analytics = self.session.transpile_cloud(
                logged,
                group="model",
            )
            distributions, job = self.session.run_cloud(
                execution_circuits,
                mode=MitigationMode.RAW,
                group="model",
                job_name="Q-ErrorID trained QNN inference",
            )
            execution_source = f"haiqu_cloud:{self.session.selected_device.id}"
        else:
            execution_circuits, distributions, analytics = (
                self.session.local_transpile_and_run(
                    circuits,
                    physical_qubits=qubits,
                    tree_edges=edges,
                )
            )
            job = None
            execution_source = f"local_backend:{self.session.selected_device.id}"

        estimates: dict[str, ChannelEstimate] = {}
        rows: list[dict[str, Any]] = []
        job_info = _job_metadata(job)
        for (channel_key, family, features, circuit), distribution in zip(
            contexts,
            distributions,
        ):
            measured_distribution = dict(distribution)
            readout_audit: dict[str, float] = {}
            if readout_calibration is not None:
                measured_distribution, readout_audit = (
                    readout_calibration.correct_joint(
                        measured_distribution,
                        tuple(
                            int(qubit)
                            for qubit in (circuit.metadata or {})[
                                "physical_qubits"
                            ]
                        ),
                    )
                )
            estimate, quantum_features = self.models.predict_qnn_distribution(
                family,
                measured_distribution,
                execution_source=execution_source,
            )
            exact = self.models.predict(family, features, model_kind="qnn")
            labels = ONE_Q_LABELS if family == "1q" else TWO_Q_LABELS
            measured_vector = np.asarray(
                [
                    *[estimate.alpha[label] for label in labels],
                    *[estimate.gamma[label] for label in labels],
                    *(
                        [estimate.kappa_down.get("down", 0.0)]
                        if family == "1q"
                        else []
                    ),
                ]
            )
            exact_vector = np.asarray(
                [
                    *[exact.alpha[label] for label in labels],
                    *[exact.gamma[label] for label in labels],
                    *(
                        [exact.kappa_down.get("down", 0.0)]
                        if family == "1q"
                        else []
                    ),
                ]
            )
            prediction = predict_features_with_agent1(estimate, family)
            held_out_rmse = (
                float(np.sqrt(np.mean((prediction - features) ** 2)))
                if prediction is not None and prediction.shape == features.shape
                else None
            )
            estimates[channel_key] = estimate
            rows.append(
                {
                    "channel_key": channel_key,
                    "features": _json(features),
                    "predicted_alpha": _json(estimate.alpha),
                    "predicted_gamma": _json(estimate.gamma),
                    "predicted_kappa_down": _json(estimate.kappa_down),
                    "held_out_prediction_rmse": held_out_rmse,
                    "model_kind": "qnn",
                    "feature_regime": "readout_corrected",
                    "feature_stage": "readout_mitigated",
                    "inference_backend": execution_source,
                    "model_source": estimate.model_source,
                    "trained_model": estimate.trained_model,
                    "qnn_circuit_name": circuit.name,
                    "qnn_observables": _json(quantum_features),
                    "parameter_mae_vs_exact_qnn": float(
                        np.mean(np.abs(measured_vector - exact_vector))
                    ),
                    **job_info,
                    **readout_audit,
                }
            )
        analytics = analytics.copy()
        analytics["model_kind"] = "qnn"
        analytics["inference_backend"] = execution_source
        analytics["status"] = (
            "executed_haiqu_cloud"
            if self.session.cloud_enabled
            else "executed_local_fallback"
        )
        return estimates, rows, analytics

    def _synthetic_ground_truth_track(self) -> pd.DataFrame:
        rows = []
        model_kinds = (
            ("ridge", "qnn")
            if self.config.model_mode == "compare"
            else (self.config.model_mode,)
        )
        families = {
            "1q": self.config.data_root / "1q_mixed_channel_test.npz",
            "2q": self.config.data_root / "2q_mixed_channel_test.npz",
        }
        for family, path in families.items():
            if not path.exists():
                rows.append(
                    {
                        "track": "synthetic_ground_truth",
                        "family": family,
                        "status": "dataset_unavailable",
                        "dataset": _portable_path(path, self.root),
                    }
                )
                continue
            with np.load(path, allow_pickle=False) as archive:
                feature_key = (
                    f"X_shot_{self.config.shots}"
                    if f"X_shot_{self.config.shots}" in archive.files
                    else "X_exact"
                )
                x = np.asarray(archive[feature_key], dtype=float)
                y_alpha = np.asarray(archive["y_alpha"], dtype=float)
                y_gamma = np.asarray(archive["y_gamma"], dtype=float)
                y_kappa = (
                    np.asarray(archive["y_kappa"], dtype=float)
                    if family == "1q" and "y_kappa" in archive.files
                    else None
                )
            for model_kind in model_kinds:
                for sample_index in range(min(len(x), 64)):
                    estimate = self.models.predict(
                        family,
                        x[sample_index],
                        model_kind=model_kind,
                    )
                    labels = (
                        ONE_Q_LABELS if family == "1q" else TWO_Q_LABELS
                    )
                    predicted_alpha = np.asarray(
                        [estimate.alpha[label] for label in labels]
                    )
                    predicted_gamma = np.asarray(
                        [estimate.gamma[label] for label in labels]
                    )
                    row = {
                        "track": "synthetic_ground_truth",
                        "family": family,
                        "model_kind": model_kind,
                        "inference_backend": (
                            "numpy_exact_statevector"
                            if model_kind == "qnn"
                            else "classical_cpu"
                        ),
                        "status": "evaluated",
                        "dataset": _portable_path(path, self.root),
                        "sample_index": sample_index,
                        "alpha_mae": float(
                            np.mean(
                                np.abs(predicted_alpha - y_alpha[sample_index])
                            )
                        ),
                        "gamma_mae": float(
                            np.mean(
                                np.abs(predicted_gamma - y_gamma[sample_index])
                            )
                        ),
                        "true_alpha": _json(y_alpha[sample_index]),
                        "predicted_alpha": _json(predicted_alpha),
                        "true_gamma": _json(y_gamma[sample_index]),
                        "predicted_gamma": _json(predicted_gamma),
                        "model_source": estimate.model_source,
                    }
                    if y_kappa is not None:
                        row["kappa_mae"] = abs(
                            estimate.kappa_down.get("down", 0.0)
                            - float(y_kappa[sample_index, 0])
                        )
                    rows.append(row)
        return pd.DataFrame(rows)

    def _mitigation_modes(
        self,
        execution_circuits: Sequence[Any],
        original_circuits: Sequence[QuantumCircuit],
        ideal_batch: FeatureBatch,
        raw_estimates: dict[str, ChannelEstimate],
        raw_rows: list[dict[str, Any]],
        calibrated_rows: list[dict[str, Any]],
        primary_model_kind: str,
        raw_job: Any | None,
    ) -> pd.DataFrame:
        all_rows = []
        raw_job_info = _job_metadata(raw_job)
        for row in raw_rows:
            all_rows.append(
                {
                    "track": "empirical_device_characterization",
                    "mode": MitigationMode.RAW.value,
                    "status": "executed",
                    "device_id": self.session.selected_device.id,
                    "shots": self.config.shots,
                    **row,
                    **raw_job_info,
                    "interpretation": (
                        "raw characterization retained as an unmitigated audit; "
                        "not used as the physical-generator fit input"
                    ),
                }
            )
        for row in calibrated_rows:
            all_rows.append(
                {
                    "track": "empirical_device_characterization",
                    "mode": "calibrated_readout",
                    "status": "executed",
                    "device_id": self.session.selected_device.id,
                    "shots": self.config.shots,
                    **row,
                    "job_id": None,
                    "qpu_cost": None,
                    "interpretation": (
                        "independent assignment calibration applied before "
                        "physical-generator inference"
                    ),
                }
            )
        for mode in list(MitigationMode)[1:]:
            if not self.session.cloud_enabled:
                for channel_key in raw_estimates:
                    all_rows.append(
                        {
                            "track": "empirical_device_characterization",
                            "mode": mode.value,
                            "status": "not_run_requires_haiqu_cloud",
                            "device_id": self.session.selected_device.id,
                            "shots": self.config.shots,
                            "channel_key": channel_key,
                            "features": None,
                            "predicted_alpha": None,
                            "predicted_gamma": None,
                            "predicted_kappa_down": None,
                            "held_out_prediction_rmse": None,
                            "model_source": None,
                            "trained_model": None,
                            "job_id": None,
                            "qpu_cost": None,
                            "interpretation": (
                                "No local result is substituted for Haiqu mitigation"
                            ),
                        }
                    )
                continue
            try:
                results, job = self.session.run_cloud(
                    execution_circuits,
                    mode=mode,
                    group="mitigation",
                    job_name=f"Q-ErrorID diagnostics / {mode.value}",
                )
                batch = results_to_features(
                    original_circuits,
                    results,
                    mode=mode.value,
                )
                _, estimated_rows = self._estimate_batch(
                    batch,
                    ideal_batch,
                    model_kind=primary_model_kind,
                    inference_backend=(
                        "numpy_exact_statevector_postprocessor"
                        if primary_model_kind == "qnn"
                        else "classical_cpu"
                    ),
                    feature_regime=(
                        "readout_corrected"
                        if mode
                        in {
                            MitigationMode.DEFAULT,
                            MitigationMode.READOUT,
                            MitigationMode.ADVANCED,
                        }
                        else "raw_shot"
                    ),
                    feature_stage=f"haiqu_{mode.value}",
                )
                job_info = _job_metadata(job)
                for row in estimated_rows:
                    all_rows.append(
                        {
                            "track": "empirical_device_characterization",
                            "mode": mode.value,
                            "status": "executed",
                            "device_id": self.session.selected_device.id,
                            "shots": self.config.shots,
                            **row,
                            **job_info,
                            "interpretation": (
                                "mitigated comparison only; not used as raw-channel truth"
                            ),
                        }
                    )
                self.session.log_object(
                    batch.table,
                    name=f"features / {mode.value}",
                    group="mitigation",
                )
            # Cloud backends can surface provider-specific exception classes;
            # every failed mitigation mode is recorded instead of aborting raw
            # characterization and the remaining comparison table.
            except Exception as exc:  # noqa: BLE001
                for channel_key in raw_estimates:
                    all_rows.append(
                        {
                            "track": "empirical_device_characterization",
                            "mode": mode.value,
                            "status": "unsupported_or_failed",
                            "device_id": self.session.selected_device.id,
                            "shots": self.config.shots,
                            "channel_key": channel_key,
                            "features": None,
                            "predicted_alpha": None,
                            "predicted_gamma": None,
                            "predicted_kappa_down": None,
                            "held_out_prediction_rmse": None,
                            "model_source": None,
                            "trained_model": None,
                            "job_id": None,
                            "qpu_cost": None,
                            "interpretation": str(exc),
                        }
                    )
        return pd.DataFrame(all_rows)

    @staticmethod
    def _edge_channel_key(
        atlas: DeviceErrorAtlas,
        edge: tuple[int, int],
    ) -> str:
        """Resolve an oriented physical edge without silently swapping CX roles."""

        requested = tuple(int(q) for q in edge)
        for edge_key, channel in atlas.two_qubit_channels.items():
            if tuple(int(q) for q in channel["physical_qubits"]) == requested:
                return edge_key
        raise KeyError(f"The error atlas has no oriented CX channel for {requested}")

    def _build_grover_benchmark_suites(
        self,
        atlas: DeviceErrorAtlas,
        qubits: tuple[int, ...],
        edges: tuple[tuple[int, int], ...],
    ) -> tuple[
        list[QuantumCircuit],
        dict[str, GeneratorResponseModel],
        bool,
        str | None,
    ]:
        """Build all targets on all selected edges before any benchmark run."""

        compact_index = {
            int(physical): logical
            for logical, physical in enumerate(qubits)
        }
        edge_specs = []
        raw_circuits: list[QuantumCircuit] = []
        for edge in edges:
            edge_key = self._edge_channel_key(atlas, edge)
            physical = tuple(int(q) for q in edge)
            circuit_qubits = tuple(compact_index[q] for q in physical)
            single_channels = (
                atlas.single_qubit_channels[physical[0]],
                atlas.single_qubit_channels[physical[1]],
            )
            two_channel = atlas.two_qubit_channels[edge_key]
            edge_specs.append(
                (
                    edge_key,
                    physical,
                    circuit_qubits,
                    single_channels,
                    two_channel,
                )
            )
            for two_qubit_depth in self.config.benchmark_two_qubit_depths:
                for target in GROVER_TARGETS:
                    circuit = build_grover_search_circuit(
                        target,
                        width=len(qubits),
                        circuit_qubits=circuit_qubits,
                        physical_qubits=physical,
                        edge_key=edge_key,
                        two_qubit_depth=two_qubit_depth,
                    )
                    circuit.metadata["response_key"] = (
                        f"{edge_key}@d2q{two_qubit_depth}"
                    )
                    raw_circuits.append(circuit)

        correction_ready = all(
            bool(channel.get("trained_model"))
            for channel in (
                list(atlas.single_qubit_channels.values())
                + list(atlas.two_qubit_channels.values())
            )
        )
        if not correction_ready:
            return (
                raw_circuits,
                {},
                False,
                "at least one atlas channel is not backed by a trained model",
            )

        response_models: dict[str, GeneratorResponseModel] = {}
        try:
            for (
                edge_key,
                _physical,
                _circuit_qubits,
                single_channels,
                two_channel,
            ) in edge_specs:
                for two_qubit_depth in self.config.benchmark_two_qubit_depths:
                    response_key = f"{edge_key}@d2q{two_qubit_depth}"
                    response_models[response_key] = (
                        GeneratorResponseModel.from_channels(
                            edge_key=edge_key,
                            single_qubit_channels=single_channels,
                            two_qubit_channel=two_channel,
                            regularization=self.config.response_regularization,
                            coherent_compensation_in_forward_circuit=False,
                            two_qubit_depth=two_qubit_depth,
                        )
                    )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return raw_circuits, {}, False, f"{type(exc).__name__}: {exc}"
        return raw_circuits, response_models, True, None

    def _benchmark_v04(
        self,
        atlas: DeviceErrorAtlas,
        qubits: tuple[int, ...],
        edges: tuple[tuple[int, int], ...],
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
        float | None,
    ]:
        """Run an exhaustive 2Q Grover benchmark through the selected runtime."""

        (
            raw_circuits,
            response_models,
            correction_ready,
            correction_unavailable_reason,
        ) = self._build_grover_benchmark_suites(atlas, qubits, edges)
        ideal_distributions = _ideal_counts(
            raw_circuits,
            shots=self.config.shots,
            seed=self.config.seed,
        )

        executed: dict[str, dict[str, Any]] = {
            "ideal_simulation": {
                "distributions": ideal_distributions,
                "circuits": raw_circuits,
                "job": None,
                "status": "executed",
            }
        }
        analytics_frames: list[pd.DataFrame] = []
        if self.session.cloud_enabled:
            raw_logged = self.session.log_circuits(raw_circuits, group="final")
            raw_execution, raw_analytics = self.session.transpile_cloud(
                raw_logged,
                group="final",
            )
            raw_analytics["benchmark_variant"] = "raw_algorithm"
            analytics_frames.append(raw_analytics)
            cloud_specs = [
                (
                    "raw_haiqu_execution",
                    raw_execution,
                    raw_circuits,
                    MitigationMode.RAW,
                ),
                (
                    "haiqu_mitigation_only",
                    raw_execution,
                    raw_circuits,
                    MitigationMode.DEFAULT,
                ),
            ]
            if correction_ready:
                cloud_specs.append(
                    (
                        "learned_full_correction_plus_haiqu_mitigation",
                        raw_execution,
                        raw_circuits,
                        MitigationMode.ADVANCED,
                    )
                )
            for label, execution_circuits, original_circuits, mode in cloud_specs:
                result, job = self.session.run_cloud(
                    execution_circuits,
                    mode=mode,
                    group="final",
                    job_name=f"Q-ErrorID Grover benchmark / {label}",
                )
                executed[label] = {
                    "distributions": [dict(item) for item in result],
                    "circuits": original_circuits,
                    "job": job,
                    "status": "executed",
                }
            if correction_ready:
                raw_record = executed["raw_haiqu_execution"]
                executed["learned_generator_correction_only"] = {
                    **raw_record,
                    "distributions": [
                        dict(item) for item in raw_record["distributions"]
                    ],
                }
        else:
            raw_transpiled, raw_counts, raw_analytics = (
                self.session.local_transpile_and_run(
                    raw_circuits,
                    physical_qubits=qubits,
                    tree_edges=edges,
                )
            )
            raw_analytics["benchmark_variant"] = "raw_algorithm"
            analytics_frames.append(raw_analytics)
            executed["raw_haiqu_execution"] = {
                "distributions": raw_counts,
                "circuits": raw_transpiled,
                "metadata_circuits": raw_circuits,
                "job": None,
                "status": "executed_local_fallback_not_haiqu",
            }
            if correction_ready:
                executed["learned_generator_correction_only"] = {
                    "distributions": [dict(item) for item in raw_counts],
                    "circuits": raw_transpiled,
                    "metadata_circuits": raw_circuits,
                    "job": None,
                    "status": "executed_local_fallback_not_haiqu",
                }

        order = (
            "ideal_simulation",
            "raw_haiqu_execution",
            "learned_generator_correction_only",
            "haiqu_mitigation_only",
            "learned_full_correction_plus_haiqu_mitigation",
        )
        inverse_scenarios = {
            "learned_generator_correction_only",
            "learned_full_correction_plus_haiqu_mitigation",
        }
        detail_rows: list[dict[str, Any]] = []
        for label in order:
            record = executed.get(label)
            if record is None:
                if label in inverse_scenarios and not correction_ready:
                    missing_status = "not_run_learned_correction_unavailable"
                    interpretation = correction_unavailable_reason
                else:
                    missing_status = "not_run_requires_haiqu_cloud"
                    interpretation = (
                        "No local result is substituted for Haiqu mitigation"
                    )
                for circuit in raw_circuits:
                    metadata = circuit.metadata or {}
                    detail_rows.append(
                        {
                            "algorithm": "grover_2q_exhaustive",
                            "scenario": label,
                            "status": missing_status,
                            "device_id": self.session.selected_device.id,
                            "shots": self.config.shots,
                            "edge_key": metadata["edge_key"],
                            "physical_qubits": metadata["physical_qubits"],
                            "target_state": metadata["target_state"],
                            "target_bitstring": f"b{metadata['target_state']}",
                            "tvd_to_ideal": None,
                            "success_probability": None,
                            "hellinger_fidelity": None,
                            "distribution": None,
                            "measured_distribution_before_inverse": None,
                            "correction_components": None,
                            "interpretation": interpretation,
                        }
                    )
                continue
            distributions = record["distributions"]
            metadata_circuits = record.get("metadata_circuits", record["circuits"])
            if len(distributions) != len(raw_circuits):
                raise RuntimeError(
                    f"{label} returned {len(distributions)} distributions for "
                    f"{len(raw_circuits)} Grover circuits"
                )
            job_info = _job_metadata(record["job"])
            for index, (distribution, metadata_circuit) in enumerate(
                zip(distributions, metadata_circuits)
            ):
                metadata = metadata_circuit.metadata or {}
                edge_key = str(metadata["edge_key"])
                target = str(metadata["target_state"])
                measured_distribution = dict(distribution)
                evaluated_distribution = measured_distribution
                response = (
                    response_models.get(
                        str(
                            metadata.get(
                                "response_key",
                                f"{edge_key}@d2q2",
                            )
                        )
                    )
                    if label in inverse_scenarios
                    else None
                )
                row_status = record["status"]
                interpretation = "direct measured distribution"
                inverse_audit: dict[str, float] = {}
                if label in inverse_scenarios:
                    if response is None or not response.validation_passed:
                        row_status = "not_applied_response_validation_failed"
                        evaluated_distribution = None
                        interpretation = (
                            "The learned non-CPTP inverse failed its forward-model "
                            "validation and was not applied"
                        )
                    else:
                        (
                            evaluated_distribution,
                            inverse_audit,
                        ) = response.correct_with_audit(
                            measured_distribution,
                            require_validation=True,
                        )
                        interpretation = (
                            "regularized full alpha/gamma/kappa algorithm-response "
                            "inversion with no added physical correction gates"
                        )

                ideal = ideal_distributions[index]
                circuit_analytics = local_circuit_analytics(
                    metadata_circuit,
                    stage="algorithm_benchmark_pretranspile",
                    device_id=self.session.selected_device.id,
                    physical_qubits=metadata["physical_qubits"],
                )
                detail_rows.append(
                    {
                        "algorithm": "grover_2q_exhaustive",
                        "scenario": label,
                        "status": row_status,
                        "device_id": (
                            "ideal_aer"
                            if label == "ideal_simulation"
                            else self.session.selected_device.id
                        ),
                        "shots": self.config.shots,
                        "edge_key": edge_key,
                        "physical_qubits": metadata["physical_qubits"],
                        "target_state": target,
                        "target_bitstring": f"b{target}",
                        "tvd_to_ideal": (
                            total_variation_distance(evaluated_distribution, ideal)
                            if evaluated_distribution is not None
                            else None
                        ),
                        "success_probability": (
                            success_probability(evaluated_distribution, [target])
                            if evaluated_distribution is not None
                            else None
                        ),
                        "hellinger_fidelity": (
                            hellinger_fidelity(evaluated_distribution, ideal)
                            if evaluated_distribution is not None
                            else None
                        ),
                        "logical_gate_count": metadata["logical_gate_count"],
                        "logical_two_qubit_gate_count": metadata[
                            "logical_two_qubit_gate_count"
                        ],
                        "coherent_correction_locations": metadata[
                            "coherent_correction_locations"
                        ],
                        "pretranspile_depth": circuit_analytics["depth"],
                        "pretranspile_two_qubit_depth": circuit_analytics[
                            "two_qubit_depth"
                        ],
                        "pretranspile_two_qubit_gate_count": circuit_analytics[
                            "two_qubit_gate_count"
                        ],
                        "response_condition_number": (
                            response.condition_number if response is not None else None
                        ),
                        "response_inverse_overhead_l1": (
                            response.inverse_overhead_l1
                            if response is not None
                            else None
                        ),
                        "response_validation_passed": (
                            response.validation_passed
                            if response is not None
                            else None
                        ),
                        "inverse_raw_normalization": inverse_audit.get(
                            "inverse_raw_normalization"
                        ),
                        "inverse_raw_negativity": inverse_audit.get(
                            "inverse_raw_negativity"
                        ),
                        "simplex_projection_l1": inverse_audit.get(
                            "simplex_projection_l1"
                        ),
                        "correction_components": (
                            "alpha,gamma,kappa_down"
                            if label in inverse_scenarios
                            else "haiqu_managed"
                            if label == "haiqu_mitigation_only"
                            else "none"
                        ),
                        "distribution": (
                            _json(normalize_distribution(evaluated_distribution))
                            if evaluated_distribution is not None
                            else None
                        ),
                        "measured_distribution_before_inverse": (
                            _json(normalize_distribution(measured_distribution))
                            if label in inverse_scenarios
                            else None
                        ),
                        "job_id": job_info["job_id"],
                        "qpu_cost": _json(job_info["qpu_cost"]),
                        "interpretation": interpretation,
                    }
                )

        details = pd.DataFrame(detail_rows)
        summary_rows = []
        for label in order:
            group = details[details["scenario"] == label]
            measured = group[
                group["status"].fillna("").str.startswith("executed")
            ]
            if measured.empty:
                summary_rows.append(
                    {
                        "algorithm": "grover_2q_exhaustive",
                        "scenario": label,
                        "status": (
                            str(group["status"].iloc[0])
                            if not group.empty
                            else "not_run"
                        ),
                        "device_id": self.session.selected_device.id,
                        "shots": self.config.shots,
                        "problem_count": len(group),
                        "edge_count": len(edges),
                        "tvd_to_ideal": None,
                        "tvd_std": None,
                        "success_probability": None,
                        "success_probability_std": None,
                        "minimum_success_probability": None,
                        "hellinger_fidelity": None,
                        "two_qubit_depth": None,
                        "two_qubit_gate_count": None,
                        "inverse_raw_negativity": None,
                        "simplex_projection_l1": None,
                        "correction_components": None,
                        "qpu_cost": None,
                    }
                )
                continue
            summary_rows.append(
                {
                    "algorithm": "grover_2q_exhaustive",
                    "scenario": label,
                    "status": str(measured["status"].iloc[0]),
                    "device_id": str(measured["device_id"].iloc[0]),
                    "shots": self.config.shots,
                    "problem_count": len(measured),
                    "edge_count": int(measured["edge_key"].nunique()),
                    "target_states": _json(list(GROVER_TARGETS)),
                    "tvd_to_ideal": float(measured["tvd_to_ideal"].mean()),
                    "tvd_std": float(measured["tvd_to_ideal"].std(ddof=0)),
                    "success_probability": float(
                        measured["success_probability"].mean()
                    ),
                    "success_probability_std": float(
                        measured["success_probability"].std(ddof=0)
                    ),
                    "minimum_success_probability": float(
                        measured["success_probability"].min()
                    ),
                    "hellinger_fidelity": float(
                        measured["hellinger_fidelity"].mean()
                    ),
                    "two_qubit_depth": float(
                        measured["pretranspile_two_qubit_depth"].mean()
                    ),
                    "two_qubit_gate_count": float(
                        measured["pretranspile_two_qubit_gate_count"].mean()
                    ),
                    "coherent_correction_locations": float(
                        measured["coherent_correction_locations"].mean()
                    ),
                    "response_condition_number": (
                        float(measured["response_condition_number"].mean())
                        if measured["response_condition_number"].notna().any()
                        else None
                    ),
                    "response_inverse_overhead_l1": (
                        float(measured["response_inverse_overhead_l1"].mean())
                        if measured["response_inverse_overhead_l1"].notna().any()
                        else None
                    ),
                    "response_validation_fraction": (
                        float(
                            measured["response_validation_passed"]
                            .dropna()
                            .astype(float)
                            .mean()
                        )
                        if measured["response_validation_passed"].notna().any()
                        else None
                    ),
                    "inverse_raw_negativity": (
                        float(measured["inverse_raw_negativity"].mean())
                        if "inverse_raw_negativity" in measured
                        and measured["inverse_raw_negativity"].notna().any()
                        else None
                    ),
                    "simplex_projection_l1": (
                        float(measured["simplex_projection_l1"].mean())
                        if "simplex_projection_l1" in measured
                        and measured["simplex_projection_l1"].notna().any()
                        else None
                    ),
                    "correction_components": str(
                        measured["correction_components"].iloc[0]
                    ),
                    "qpu_cost": str(measured["qpu_cost"].iloc[0]),
                }
            )
        summary = pd.DataFrame(summary_rows)
        raw = summary.loc[
            summary["scenario"] == "raw_haiqu_execution", "tvd_to_ideal"
        ].iloc[0]
        corrected_value = summary.loc[
            summary["scenario"] == "learned_generator_correction_only",
            "tvd_to_ideal",
        ].iloc[0]
        improvement = (
            float(raw - corrected_value)
            if pd.notna(raw) and pd.notna(corrected_value)
            else None
        )
        response_payload = {
            "algorithm": "grover_2q_exhaustive",
            "targets": list(GROVER_TARGETS),
            "uses_benchmark_counts_for_response_fit": False,
            "correction_stack": {
                "alpha": (
                    "propagated after every modeled gate and included in the "
                    "regularized algorithm-response inverse"
                ),
                "gamma": "included in the regularized algorithm-response inverse",
                "kappa_down": (
                    "non-unital component included in the same response inverse"
                ),
                "haiqu": "optional SDK mitigation layer",
                "added_physical_correction_gates": 0,
            },
            "semantics": (
                "algorithm-level error cancellation; not fault-tolerant QEC"
            ),
            "models": {
                edge_key: model.to_dict()
                for edge_key, model in response_models.items()
            },
            "correction_ready": correction_ready,
            "correction_unavailable_reason": correction_unavailable_reason,
        }
        benchmark_analytics = (
            pd.concat(analytics_frames, ignore_index=True)
            if analytics_frames
            else pd.DataFrame()
        )
        self.session.log_object(
            summary,
            name="Grover algorithm benchmark summary",
            group="final",
        )
        self.session.log_object(
            details,
            name="Grover algorithm benchmark instances",
            group="final",
        )
        self.session.log_object(
            response_payload,
            name="learned generator response models",
            group="final",
        )
        return (
            summary,
            details,
            benchmark_analytics,
            response_payload,
            improvement,
        )

    def _benchmark(
        self,
        atlas: DeviceErrorAtlas,
        qubits: tuple[int, ...],
        edges: tuple[tuple[int, int], ...],
        readout_calibration: ReadoutCalibrationBundle | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        dict[str, Any],
        float | None,
    ]:
        """Run held-out, multi-seed v0.5 Grover mitigation validation."""

        if readout_calibration is None:
            readout_calibration = ReadoutCalibrationBundle.identity(qubits, edges)
        (
            raw_circuits,
            response_models,
            correction_ready,
            correction_unavailable_reason,
        ) = self._build_grover_benchmark_suites(atlas, qubits, edges)
        ideal_distributions = _ideal_counts(
            raw_circuits,
            shots=self.config.shots,
            seed=self.config.seed,
        )
        circuit_count = len(raw_circuits)
        validation_seeds = [
            self.config.seed + 10_001 + index
            for index in range(self.config.validation_repeats)
        ]
        evaluation_seeds = [
            self.config.seed + 20_001 + index
            for index in range(self.config.evaluation_repeats)
        ]

        analytics_frames: list[pd.DataFrame] = []
        default_evaluation_runs: list[list[dict[str, int | float]]] | None = None
        advanced_evaluation_runs: list[list[dict[str, int | float]]] | None = None
        default_job = None
        advanced_job = None
        if self.session.cloud_enabled:
            raw_logged = self.session.log_circuits(raw_circuits, group="final")
            raw_execution, raw_analytics = self.session.transpile_cloud(
                raw_logged,
                group="final",
            )
            raw_analytics["benchmark_variant"] = "raw_algorithm_level_2"
            analytics_frames.append(raw_analytics)
            total_repeats = (
                self.config.validation_repeats + self.config.evaluation_repeats
            )
            raw_flat, raw_job = self.session.run_cloud(
                list(raw_execution) * total_repeats,
                mode=MitigationMode.RAW,
                group="final",
                job_name="Q-ErrorID v0.5 Grover / raw repeated",
            )
            all_raw_runs = _split_repeated_results(
                raw_flat,
                circuit_count=circuit_count,
                repeats=total_repeats,
            )
            validation_runs = all_raw_runs[: self.config.validation_repeats]
            evaluation_runs = all_raw_runs[self.config.validation_repeats :]

            default_flat, default_job = self.session.run_cloud(
                list(raw_execution) * self.config.evaluation_repeats,
                mode=MitigationMode.DEFAULT,
                group="final",
                job_name="Q-ErrorID v0.5 Grover / Haiqu mitigation repeated",
            )
            default_evaluation_runs = _split_repeated_results(
                default_flat,
                circuit_count=circuit_count,
                repeats=self.config.evaluation_repeats,
            )
            if correction_ready:
                advanced_flat, advanced_job = self.session.run_cloud(
                    list(raw_execution) * self.config.evaluation_repeats,
                    mode=MitigationMode.ADVANCED,
                    group="final",
                    job_name=(
                        "Q-ErrorID v0.5 Grover / learned plus Haiqu repeated"
                    ),
                )
                advanced_evaluation_runs = _split_repeated_results(
                    advanced_flat,
                    circuit_count=circuit_count,
                    repeats=self.config.evaluation_repeats,
                )
            raw_status = "executed_haiqu_cloud"
        else:
            (
                raw_transpiled,
                first_validation_counts,
                raw_analytics,
            ) = self.session.local_transpile_and_run(
                raw_circuits,
                physical_qubits=qubits,
                tree_edges=edges,
                seed=validation_seeds[0],
            )
            raw_analytics["benchmark_variant"] = "raw_algorithm_level_2"
            analytics_frames.append(raw_analytics)
            if self.session.local_runtime is None:
                raise RuntimeError("Local runtime is not initialized")
            validation_runs = [[dict(item) for item in first_validation_counts]]
            for seed in validation_seeds[1:]:
                validation_runs.append(
                    [
                        dict(item)
                        for item in self.session.local_runtime.run(
                            raw_transpiled,
                            shots=self.config.shots,
                            seed=seed,
                        )
                    ]
                )
            evaluation_runs = [
                [
                    dict(item)
                    for item in self.session.local_runtime.run(
                        raw_transpiled,
                        shots=self.config.shots,
                        seed=seed,
                    )
                ]
                for seed in evaluation_seeds
            ]
            raw_job = None
            raw_status = "executed_local_fallback_not_haiqu"

        validation_rows: list[dict[str, Any]] = []
        validation_repeat_improvements: list[float] = []
        for repeat_index, (seed, distributions) in enumerate(
            zip(validation_seeds, validation_runs)
        ):
            repeat_improvements = []
            for instance_index, (distribution, ideal) in enumerate(
                zip(distributions, ideal_distributions)
            ):
                metadata = raw_circuits[instance_index].metadata or {}
                edge_key = str(metadata["edge_key"])
                response_key = str(metadata["response_key"])
                two_qubit_depth = int(metadata["requested_two_qubit_depth"])
                target = str(metadata["target_state"])
                readout_distribution, readout_audit = (
                    readout_calibration.correct_edge(edge_key, distribution)
                )
                raw_tvd = total_variation_distance(distribution, ideal)
                readout_tvd = total_variation_distance(
                    readout_distribution,
                    ideal,
                )
                candidate_tvd = None
                improvement = None
                response = response_models.get(response_key)
                response_valid = bool(
                    response is not None and response.validation_passed
                )
                if correction_ready and response_valid:
                    candidate, inverse_audit = response.correct_with_audit(
                        readout_distribution
                    )
                    candidate_tvd = total_variation_distance(candidate, ideal)
                    improvement = readout_tvd - candidate_tvd
                    repeat_improvements.append(improvement)
                else:
                    inverse_audit = {}
                validation_rows.append(
                    {
                        "split": "validation",
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "instance_index": instance_index,
                        "edge_key": edge_key,
                        "response_key": response_key,
                        "requested_two_qubit_depth": two_qubit_depth,
                        "target_state": target,
                        "raw_tvd": raw_tvd,
                        "readout_only_tvd": readout_tvd,
                        "candidate_full_stack_tvd": candidate_tvd,
                        "incremental_generator_improvement_tvd": improvement,
                        "response_validation_passed": response_valid,
                        **readout_audit,
                        **inverse_audit,
                    }
                )
            if repeat_improvements:
                validation_repeat_improvements.append(
                    float(np.mean(repeat_improvements))
                )

        (
            validation_mean,
            validation_ci_low,
            validation_ci_high,
        ) = _mean_ci95(validation_repeat_improvements)
        positive_fraction = (
            float(
                np.mean(np.asarray(validation_repeat_improvements, dtype=float) > 0.0)
            )
            if validation_repeat_improvements
            else 0.0
        )
        statistically_significant = bool(
            validation_ci_low is not None and validation_ci_low > 0.0
        )
        all_response_models_valid = bool(
            correction_ready
            and response_models
            and all(model.validation_passed for model in response_models.values())
        )
        validation_negativity = np.asarray(
            [
                float(row["inverse_raw_negativity"])
                for row in validation_rows
                if row.get("inverse_raw_negativity") is not None
            ],
            dtype=float,
        )
        validation_projection = np.asarray(
            [
                float(row["simplex_projection_l1"])
                for row in validation_rows
                if row.get("simplex_projection_l1") is not None
            ],
            dtype=float,
        )
        mean_validation_negativity = (
            float(validation_negativity.mean())
            if validation_negativity.size
            else float("inf")
        )
        mean_validation_projection = (
            float(validation_projection.mean())
            if validation_projection.size
            else float("inf")
        )
        audit_safe = bool(
            mean_validation_negativity <= 0.05
            and mean_validation_projection <= 0.10
        )
        edge_depth_validation: dict[str, dict[str, Any]] = {}
        for response_key in sorted(
            {
                str((circuit.metadata or {})["response_key"])
                for circuit in raw_circuits
            }
        ):
            matching = next(
                circuit
                for circuit in raw_circuits
                if str((circuit.metadata or {})["response_key"])
                == response_key
            )
            matching_metadata = matching.metadata or {}
            edge_key = str(matching_metadata["edge_key"])
            two_qubit_depth = int(
                matching_metadata["requested_two_qubit_depth"]
            )
            edge_repeat_improvements: list[float] = []
            for repeat_index in range(self.config.validation_repeats):
                values = [
                    float(row["incremental_generator_improvement_tvd"])
                    for row in validation_rows
                    if row["response_key"] == response_key
                    and row["repeat_index"] == repeat_index
                    and row.get("incremental_generator_improvement_tvd")
                    is not None
                ]
                if values:
                    edge_repeat_improvements.append(float(np.mean(values)))
            edge_mean, edge_ci_low, edge_ci_high = _mean_ci95(
                edge_repeat_improvements
            )
            edge_positive_fraction = (
                float(
                    np.mean(
                        np.asarray(edge_repeat_improvements, dtype=float) > 0.0
                    )
                )
                if edge_repeat_improvements
                else 0.0
            )
            edge_rows = [
                row
                for row in validation_rows
                if row["response_key"] == response_key
            ]
            edge_negativity = np.asarray(
                [
                    float(row["inverse_raw_negativity"])
                    for row in edge_rows
                    if row.get("inverse_raw_negativity") is not None
                ],
                dtype=float,
            )
            edge_projection = np.asarray(
                [
                    float(row["simplex_projection_l1"])
                    for row in edge_rows
                    if row.get("simplex_projection_l1") is not None
                ],
                dtype=float,
            )
            edge_mean_negativity = (
                float(edge_negativity.mean())
                if edge_negativity.size
                else float("inf")
            )
            edge_mean_projection = (
                float(edge_projection.mean())
                if edge_projection.size
                else float("inf")
            )
            response = response_models.get(response_key)
            response_valid = bool(
                correction_ready
                and response is not None
                and response.validation_passed
            )
            edge_audit_safe = bool(
                edge_mean_negativity <= 0.05
                and edge_mean_projection <= 0.10
            )
            generator_enabled = bool(
                response_valid
                and readout_calibration.validation_passed
                and np.isfinite(edge_mean)
                and edge_mean > 0.0
                and edge_positive_fraction >= 0.5
                and edge_audit_safe
            )
            edge_depth_validation[response_key] = {
                "edge_key": edge_key,
                "requested_two_qubit_depth": two_qubit_depth,
                "generator_enabled": generator_enabled,
                "fallback": (
                    "readout_only"
                    if readout_calibration.validation_passed
                    and not generator_enabled
                    else None
                ),
                "mean_incremental_tvd_improvement": (
                    edge_mean if np.isfinite(edge_mean) else None
                ),
                "ci95_low": edge_ci_low,
                "ci95_high": edge_ci_high,
                "positive_repeat_fraction": edge_positive_fraction,
                "mean_inverse_raw_negativity": (
                    edge_mean_negativity
                    if np.isfinite(edge_mean_negativity)
                    else None
                ),
                "mean_simplex_projection_l1": (
                    edge_mean_projection
                    if np.isfinite(edge_mean_projection)
                    else None
                ),
                "inverse_audit_safe": edge_audit_safe,
                "response_validation_passed": response_valid,
                "statistically_significant_at_95_percent": bool(
                    edge_ci_low is not None and edge_ci_low > 0.0
                ),
                "validation_repeat_improvements": edge_repeat_improvements,
            }
        for row in validation_rows:
            decision = edge_depth_validation[str(row["response_key"])]
            row["generator_enabled_for_evaluation"] = decision[
                "generator_enabled"
            ]
            row["selected_stack_for_evaluation"] = (
                "readout_plus_generator"
                if decision["generator_enabled"]
                else "readout_only"
            )
        enabled_edge_depths = [
            response_key
            for response_key, decision in edge_depth_validation.items()
            if decision["generator_enabled"]
        ]
        fallback_edge_depths = [
            response_key
            for response_key, decision in edge_depth_validation.items()
            if decision["fallback"] == "readout_only"
        ]
        enabled_edges = sorted(
            {
                str(decision["edge_key"])
                for decision in edge_depth_validation.values()
                if decision["generator_enabled"]
            }
        )
        fallback_edges = sorted(
            {
                str(decision["edge_key"])
                for decision in edge_depth_validation.values()
                if decision["fallback"] == "readout_only"
            }
        )
        correction_enabled = bool(enabled_edge_depths)
        validation_summary = {
            "selection_split": "independent simulator seeds or repeated cloud batches",
            "evaluation_counts_used_for_selection": False,
            "criterion": (
                "applied separately per physical edge: mean paired TVD "
                "improvement over readout-only > 0 and at least half of "
                "validation repeats improve; mean inverse negativity <= 0.05; "
                "mean simplex correction <= 0.10"
            ),
            "validation_repeats": self.config.validation_repeats,
            "validation_seeds": validation_seeds,
            "mean_incremental_tvd_improvement": (
                validation_mean if np.isfinite(validation_mean) else None
            ),
            "ci95_low": validation_ci_low,
            "ci95_high": validation_ci_high,
            "positive_repeat_fraction": positive_fraction,
            "mean_inverse_raw_negativity": (
                mean_validation_negativity
                if np.isfinite(mean_validation_negativity)
                else None
            ),
            "mean_simplex_projection_l1": (
                mean_validation_projection
                if np.isfinite(mean_validation_projection)
                else None
            ),
            "inverse_audit_safe": audit_safe,
            "statistically_significant_at_95_percent": statistically_significant,
            "readout_calibration_validation_passed": (
                readout_calibration.validation_passed
            ),
            "generator_response_models_ready": correction_ready,
            "generator_response_models_valid": all_response_models_valid,
            "correction_enabled_for_held_out_evaluation": correction_enabled,
            "generator_enabled_edges": enabled_edges,
            "readout_only_fallback_edges": fallback_edges,
            "generator_enabled_edge_depths": enabled_edge_depths,
            "readout_only_fallback_edge_depths": fallback_edge_depths,
            "per_edge_depth": edge_depth_validation,
            "failure_reason": (
                None
                if correction_enabled
                else correction_unavailable_reason
                or (
                    "empirical validation failed the paired-improvement or "
                    "inverse-audit safety criterion"
                )
            ),
        }

        detail_rows: list[dict[str, Any]] = []
        raw_analytics = raw_analytics.reset_index(drop=True)

        def append_detail(
            *,
            scenario: str,
            status: str,
            evaluated_distribution: Mapping[str, int | float] | None,
            measured_distribution: Mapping[str, int | float] | None,
            distribution_after_readout: Mapping[str, int | float] | None,
            instance_index: int,
            repeat_index: int,
            seed: int | None,
            correction_components: str,
            interpretation: str,
            response: GeneratorResponseModel | None = None,
            readout_audit: Mapping[str, float] | None = None,
            inverse_audit: Mapping[str, float] | None = None,
            generator_validation_passed: bool | None = None,
            generator_correction_applied: bool = False,
            job: Any | None = None,
        ) -> None:
            metadata_circuit = raw_circuits[instance_index]
            metadata = metadata_circuit.metadata or {}
            edge_key = str(metadata["edge_key"])
            target = str(metadata["target_state"])
            ideal = ideal_distributions[instance_index]
            pretranspile = local_circuit_analytics(
                metadata_circuit,
                stage="algorithm_benchmark_pretranspile",
                device_id=self.session.selected_device.id,
                physical_qubits=metadata["physical_qubits"],
            )
            transpiled = (
                raw_analytics.iloc[instance_index].to_dict()
                if instance_index < len(raw_analytics)
                else {}
            )
            job_info = _job_metadata(job)
            detail_rows.append(
                {
                    "algorithm": "grover_2q_exhaustive",
                    "scenario": scenario,
                    "status": status,
                    "split": "evaluation",
                    "repeat_index": repeat_index,
                    "benchmark_seed": seed,
                    "instance_index": instance_index,
                    "device_id": (
                        "ideal_aer"
                        if scenario == "ideal_simulation"
                        else self.session.selected_device.id
                    ),
                    "shots": self.config.shots,
                    "edge_key": edge_key,
                    "response_key": metadata["response_key"],
                    "requested_two_qubit_depth": metadata[
                        "requested_two_qubit_depth"
                    ],
                    "depth_sweep_method": metadata["depth_sweep_method"],
                    "physical_qubits": metadata["physical_qubits"],
                    "target_state": target,
                    "target_bitstring": f"b{target}",
                    "tvd_to_ideal": (
                        total_variation_distance(evaluated_distribution, ideal)
                        if evaluated_distribution is not None
                        else None
                    ),
                    "success_probability": (
                        success_probability(evaluated_distribution, [target])
                        if evaluated_distribution is not None
                        else None
                    ),
                    "hellinger_fidelity": (
                        hellinger_fidelity(evaluated_distribution, ideal)
                        if evaluated_distribution is not None
                        else None
                    ),
                    "logical_gate_count": metadata["logical_gate_count"],
                    "logical_two_qubit_gate_count": metadata[
                        "logical_two_qubit_gate_count"
                    ],
                    "coherent_correction_locations": metadata[
                        "coherent_correction_locations"
                    ],
                    "pretranspile_depth": pretranspile["depth"],
                    "pretranspile_two_qubit_depth": pretranspile[
                        "two_qubit_depth"
                    ],
                    "pretranspile_two_qubit_gate_count": pretranspile[
                        "two_qubit_gate_count"
                    ],
                    "transpiled_depth": transpiled.get("depth"),
                    "transpiled_two_qubit_depth": transpiled.get(
                        "two_qubit_depth"
                    ),
                    "transpiled_two_qubit_gate_count": transpiled.get(
                        "two_qubit_gate_count"
                    ),
                    "response_condition_number": (
                        response.condition_number if response is not None else None
                    ),
                    "response_inverse_overhead_l1": (
                        response.inverse_overhead_l1
                        if response is not None
                        else None
                    ),
                    "response_validation_passed": (
                        response.validation_passed
                        if response is not None
                        else None
                    ),
                    "correction_validation_passed": correction_enabled,
                    "generator_validation_passed": (
                        generator_validation_passed
                    ),
                    "generator_correction_applied": (
                        generator_correction_applied
                    ),
                    **dict(readout_audit or {}),
                    **dict(inverse_audit or {}),
                    "correction_components": correction_components,
                    "distribution": (
                        _json(normalize_distribution(evaluated_distribution))
                        if evaluated_distribution is not None
                        else None
                    ),
                    "measured_distribution_before_inverse": (
                        _json(normalize_distribution(measured_distribution))
                        if measured_distribution is not None
                        else None
                    ),
                    "distribution_after_readout": (
                        _json(normalize_distribution(distribution_after_readout))
                        if distribution_after_readout is not None
                        else None
                    ),
                    "job_id": job_info["job_id"],
                    "qpu_cost": _json(job_info["qpu_cost"]),
                    "interpretation": interpretation,
                }
            )

        for repeat_index, (seed, raw_distributions) in enumerate(
            zip(evaluation_seeds, evaluation_runs)
        ):
            for instance_index, measured_distribution in enumerate(
                raw_distributions
            ):
                metadata = raw_circuits[instance_index].metadata or {}
                edge_key = str(metadata["edge_key"])
                response_key = str(metadata["response_key"])
                response = response_models.get(response_key)
                edge_generator_enabled = bool(
                    edge_depth_validation.get(response_key, {}).get(
                        "generator_enabled",
                        False,
                    )
                )
                ideal = ideal_distributions[instance_index]
                append_detail(
                    scenario="ideal_simulation",
                    status="executed",
                    evaluated_distribution=ideal,
                    measured_distribution=None,
                    distribution_after_readout=None,
                    instance_index=instance_index,
                    repeat_index=repeat_index,
                    seed=seed,
                    correction_components="none",
                    interpretation="deterministic ideal reference",
                )
                append_detail(
                    scenario="raw_haiqu_execution",
                    status=raw_status,
                    evaluated_distribution=measured_distribution,
                    measured_distribution=measured_distribution,
                    distribution_after_readout=None,
                    instance_index=instance_index,
                    repeat_index=repeat_index,
                    seed=seed,
                    correction_components="none",
                    interpretation="raw measured distribution",
                    job=raw_job,
                )
                readout_distribution, readout_audit = (
                    readout_calibration.correct_edge(
                        edge_key,
                        measured_distribution,
                    )
                )
                append_detail(
                    scenario="calibrated_readout_mitigation_only",
                    status="executed_independent_readout_calibration",
                    evaluated_distribution=readout_distribution,
                    measured_distribution=measured_distribution,
                    distribution_after_readout=readout_distribution,
                    instance_index=instance_index,
                    repeat_index=repeat_index,
                    seed=seed,
                    correction_components="readout_assignment",
                    interpretation=(
                        "regularized inversion of an independently measured "
                        "edge assignment matrix"
                    ),
                    readout_audit=readout_audit,
                    job=raw_job,
                )

                if response is not None and response.validation_passed:
                    generator_only, generator_only_audit = (
                        response.correct_with_audit(measured_distribution)
                    )
                    generator_only_status = (
                        "executed_ablation_not_recommended"
                    )
                else:
                    generator_only = None
                    generator_only_audit = {}
                    generator_only_status = (
                        "not_run_learned_correction_unavailable"
                    )
                append_detail(
                    scenario="learned_generator_correction_only",
                    status=generator_only_status,
                    evaluated_distribution=generator_only,
                    measured_distribution=measured_distribution,
                    distribution_after_readout=None,
                    instance_index=instance_index,
                    repeat_index=repeat_index,
                    seed=seed,
                    correction_components="alpha,gamma,kappa_down",
                    interpretation=(
                        "v0.4 generator-only ablation; readout nuisance remains"
                    ),
                    response=response,
                    inverse_audit=generator_only_audit,
                    job=raw_job,
                )

                if (
                    edge_generator_enabled
                    and response is not None
                    and response.validation_passed
                ):
                    full_distribution, full_inverse_audit = (
                        response.correct_with_audit(readout_distribution)
                    )
                    full_status = "executed_validated_correction"
                    full_interpretation = (
                        "independent readout assignment inversion followed by "
                        "validated alpha/gamma/kappa algorithm-response inversion"
                    )
                    full_components = (
                        "readout_assignment,alpha,gamma,kappa_down"
                    )
                    generator_applied = True
                elif readout_calibration.validation_passed:
                    full_distribution = readout_distribution
                    full_inverse_audit = {}
                    full_status = "executed_validated_readout_fallback"
                    full_interpretation = (
                        "independent validation rejected the incremental "
                        "generator inverse for this edge; retained the "
                        "validated readout-only correction"
                    )
                    full_components = "readout_assignment"
                    generator_applied = False
                else:
                    full_distribution = None
                    full_inverse_audit = {}
                    full_status = "not_applied_readout_validation_failed"
                    full_interpretation = str(
                        validation_summary["failure_reason"]
                    )
                    full_components = "none"
                    generator_applied = False
                append_detail(
                    scenario="learned_readout_plus_generator_correction",
                    status=full_status,
                    evaluated_distribution=full_distribution,
                    measured_distribution=measured_distribution,
                    distribution_after_readout=readout_distribution,
                    instance_index=instance_index,
                    repeat_index=repeat_index,
                    seed=seed,
                    correction_components=full_components,
                    interpretation=full_interpretation,
                    response=response,
                    readout_audit=readout_audit,
                    inverse_audit=full_inverse_audit,
                    generator_validation_passed=edge_generator_enabled,
                    generator_correction_applied=generator_applied,
                    job=raw_job,
                )

                if default_evaluation_runs is None:
                    haiqu_distribution = None
                    haiqu_status = "not_run_requires_haiqu_cloud"
                else:
                    haiqu_distribution = default_evaluation_runs[
                        repeat_index
                    ][instance_index]
                    haiqu_status = "executed_haiqu_cloud"
                append_detail(
                    scenario="haiqu_mitigation_only",
                    status=haiqu_status,
                    evaluated_distribution=haiqu_distribution,
                    measured_distribution=haiqu_distribution,
                    distribution_after_readout=None,
                    instance_index=instance_index,
                    repeat_index=repeat_index,
                    seed=None if self.session.cloud_enabled else seed,
                    correction_components="haiqu_managed",
                    interpretation=(
                        "Haiqu SDK default mitigation"
                        if haiqu_distribution is not None
                        else "No local result substitutes for Haiqu mitigation"
                    ),
                    job=default_job,
                )

                if advanced_evaluation_runs is not None:
                    advanced_measured = advanced_evaluation_runs[
                        repeat_index
                    ][instance_index]
                    if edge_generator_enabled and response is not None:
                        learned_haiqu, learned_haiqu_audit = (
                            response.correct_with_audit(advanced_measured)
                        )
                        learned_haiqu_components = (
                            "haiqu_managed,alpha,gamma,kappa_down"
                        )
                        learned_haiqu_interpretation = (
                            "Haiqu advanced mitigation followed by the "
                            "edge-validated generator-response inverse; no "
                            "duplicate local readout inversion"
                        )
                        learned_haiqu_generator_applied = True
                    else:
                        learned_haiqu = advanced_measured
                        learned_haiqu_audit = {}
                        learned_haiqu_components = "haiqu_managed"
                        learned_haiqu_interpretation = (
                            "independent validation rejected the incremental "
                            "generator inverse for this edge; retained Haiqu "
                            "advanced mitigation without an extra inverse"
                        )
                        learned_haiqu_generator_applied = False
                    learned_haiqu_status = "executed_haiqu_cloud"
                else:
                    advanced_measured = None
                    learned_haiqu = None
                    learned_haiqu_audit = {}
                    learned_haiqu_components = (
                        "haiqu_managed,edge_adaptive_generator"
                    )
                    learned_haiqu_interpretation = (
                        "No local result substitutes for Haiqu mitigation"
                    )
                    learned_haiqu_generator_applied = False
                    learned_haiqu_status = (
                        "not_run_requires_haiqu_cloud"
                        if not self.session.cloud_enabled
                        else "not_applied_empirical_validation_failed"
                    )
                append_detail(
                    scenario=(
                        "learned_full_correction_plus_haiqu_mitigation"
                    ),
                    status=learned_haiqu_status,
                    evaluated_distribution=learned_haiqu,
                    measured_distribution=advanced_measured,
                    distribution_after_readout=None,
                    instance_index=instance_index,
                    repeat_index=repeat_index,
                    seed=None if self.session.cloud_enabled else seed,
                    correction_components=learned_haiqu_components,
                    interpretation=learned_haiqu_interpretation,
                    response=response,
                    inverse_audit=learned_haiqu_audit,
                    generator_validation_passed=edge_generator_enabled,
                    generator_correction_applied=(
                        learned_haiqu_generator_applied
                    ),
                    job=advanced_job,
                )

        details = pd.DataFrame(detail_rows)
        order = (
            "ideal_simulation",
            "raw_haiqu_execution",
            "calibrated_readout_mitigation_only",
            "learned_generator_correction_only",
            "learned_readout_plus_generator_correction",
            "haiqu_mitigation_only",
            "learned_full_correction_plus_haiqu_mitigation",
        )
        summary_rows: list[dict[str, Any]] = []
        seed_summary_rows: list[dict[str, Any]] = []
        for scenario in order:
            group = details[details["scenario"] == scenario]
            measured = group[
                group["status"].fillna("").str.startswith("executed")
            ]
            if measured.empty:
                summary_rows.append(
                    {
                        "algorithm": "grover_2q_exhaustive",
                        "scenario": scenario,
                        "status": (
                            str(group["status"].iloc[0])
                            if not group.empty
                            else "not_run"
                        ),
                        "device_id": self.session.selected_device.id,
                        "shots": self.config.shots,
                        "problem_count": circuit_count,
                        "sample_count": 0,
                        "repeat_count": 0,
                        "edge_count": len(edges),
                        "tvd_to_ideal": None,
                        "tvd_std": None,
                        "tvd_ci95_low": None,
                        "tvd_ci95_high": None,
                        "success_probability": None,
                        "success_probability_std": None,
                        "success_probability_ci95_low": None,
                        "success_probability_ci95_high": None,
                        "minimum_success_probability": None,
                        "hellinger_fidelity": None,
                        "transpiled_depth": None,
                        "two_qubit_depth": None,
                        "two_qubit_gate_count": None,
                        "correction_components": None,
                        "correction_validation_passed": correction_enabled,
                    }
                )
                continue

            repeat_metrics = (
                measured.groupby("repeat_index", as_index=False)
                .agg(
                    tvd_to_ideal=("tvd_to_ideal", "mean"),
                    success_probability=("success_probability", "mean"),
                    hellinger_fidelity=("hellinger_fidelity", "mean"),
                )
                .sort_values("repeat_index")
            )
            for _, repeat_row in repeat_metrics.iterrows():
                repeat_index = int(repeat_row["repeat_index"])
                seed_values = measured.loc[
                    measured["repeat_index"] == repeat_index,
                    "benchmark_seed",
                ].dropna()
                seed_summary_rows.append(
                    {
                        "scenario": scenario,
                        "repeat_index": repeat_index,
                        "seed": (
                            int(seed_values.iloc[0])
                            if not seed_values.empty
                            else None
                        ),
                        "problem_count": int(
                            (measured["repeat_index"] == repeat_index).sum()
                        ),
                        "tvd_to_ideal": float(
                            repeat_row["tvd_to_ideal"]
                        ),
                        "success_probability": float(
                            repeat_row["success_probability"]
                        ),
                        "hellinger_fidelity": float(
                            repeat_row["hellinger_fidelity"]
                        ),
                    }
                )
            tvd_mean, tvd_ci_low, tvd_ci_high = _mean_ci95(
                repeat_metrics["tvd_to_ideal"]
            )
            success_mean, success_ci_low, success_ci_high = _mean_ci95(
                repeat_metrics["success_probability"]
            )
            tvd_ci_low = (
                max(0.0, tvd_ci_low) if tvd_ci_low is not None else None
            )
            tvd_ci_high = (
                min(1.0, tvd_ci_high) if tvd_ci_high is not None else None
            )
            success_ci_low = (
                max(0.0, success_ci_low)
                if success_ci_low is not None
                else None
            )
            success_ci_high = (
                min(1.0, success_ci_high)
                if success_ci_high is not None
                else None
            )
            summary_rows.append(
                {
                    "algorithm": "grover_2q_exhaustive",
                    "scenario": scenario,
                    "status": (
                        (
                            "executed_validation_gated_learned_correction"
                            if correction_enabled
                            else "executed_readout_only_fallback_no_generator"
                        )
                        if scenario
                        == "learned_readout_plus_generator_correction"
                        else str(measured["status"].iloc[0])
                    ),
                    "device_id": str(measured["device_id"].iloc[0]),
                    "shots": self.config.shots,
                    "problem_count": int(
                        measured[
                            [
                                "edge_key",
                                "target_state",
                                "requested_two_qubit_depth",
                            ]
                        ]
                        .drop_duplicates()
                        .shape[0]
                    ),
                    "sample_count": len(measured),
                    "repeat_count": int(measured["repeat_index"].nunique()),
                    "edge_count": int(measured["edge_key"].nunique()),
                    "target_states": _json(list(GROVER_TARGETS)),
                    "requested_two_qubit_depths": _json(
                        list(self.config.benchmark_two_qubit_depths)
                    ),
                    "tvd_to_ideal": tvd_mean,
                    "tvd_std": float(measured["tvd_to_ideal"].std(ddof=0)),
                    "tvd_repeat_std": float(
                        repeat_metrics["tvd_to_ideal"].std(ddof=0)
                    ),
                    "tvd_ci95_low": tvd_ci_low,
                    "tvd_ci95_high": tvd_ci_high,
                    "success_probability": success_mean,
                    "success_probability_std": float(
                        measured["success_probability"].std(ddof=0)
                    ),
                    "success_probability_repeat_std": float(
                        repeat_metrics["success_probability"].std(ddof=0)
                    ),
                    "success_probability_ci95_low": success_ci_low,
                    "success_probability_ci95_high": success_ci_high,
                    "minimum_success_probability": float(
                        measured["success_probability"].min()
                    ),
                    "hellinger_fidelity": float(
                        measured["hellinger_fidelity"].mean()
                    ),
                    "transpiled_depth": (
                        float(measured["transpiled_depth"].dropna().mean())
                        if measured["transpiled_depth"].notna().any()
                        else None
                    ),
                    "two_qubit_depth": (
                        float(
                            measured["transpiled_two_qubit_depth"]
                            .dropna()
                            .mean()
                        )
                        if measured["transpiled_two_qubit_depth"].notna().any()
                        else None
                    ),
                    "two_qubit_gate_count": (
                        float(
                            measured["transpiled_two_qubit_gate_count"]
                            .dropna()
                            .mean()
                        )
                        if measured[
                            "transpiled_two_qubit_gate_count"
                        ].notna().any()
                        else None
                    ),
                    "response_condition_number": (
                        float(measured["response_condition_number"].mean())
                        if measured["response_condition_number"].notna().any()
                        else None
                    ),
                    "response_inverse_overhead_l1": (
                        float(measured["response_inverse_overhead_l1"].mean())
                        if measured["response_inverse_overhead_l1"].notna().any()
                        else None
                    ),
                    "readout_condition_number": (
                        float(measured["readout_condition_number"].mean())
                        if measured["readout_condition_number"].notna().any()
                        else None
                    ),
                    "readout_inverse_overhead_l1": (
                        float(measured["readout_inverse_overhead_l1"].mean())
                        if measured["readout_inverse_overhead_l1"].notna().any()
                        else None
                    ),
                    "inverse_raw_negativity": (
                        float(measured["inverse_raw_negativity"].mean())
                        if measured["inverse_raw_negativity"].notna().any()
                        else None
                    ),
                    "simplex_projection_l1": (
                        float(measured["simplex_projection_l1"].mean())
                        if measured["simplex_projection_l1"].notna().any()
                        else None
                    ),
                    "correction_components": (
                        "edge_adaptive(readout_only|readout_plus_"
                        "alpha,gamma,kappa_down)"
                        if scenario
                        == "learned_readout_plus_generator_correction"
                        else str(measured["correction_components"].iloc[0])
                    ),
                    "correction_validation_passed": (
                        readout_calibration.validation_passed
                        if scenario
                        == "learned_readout_plus_generator_correction"
                        else correction_enabled
                    ),
                    "generator_enabled_edges": (
                        _json(enabled_edges)
                        if scenario
                        == "learned_readout_plus_generator_correction"
                        else None
                    ),
                    "readout_only_fallback_edges": (
                        _json(fallback_edges)
                        if scenario
                        == "learned_readout_plus_generator_correction"
                        else None
                    ),
                    "qpu_cost": str(measured["qpu_cost"].iloc[0]),
                }
            )

        summary = pd.DataFrame(summary_rows)
        seed_summary = pd.DataFrame(seed_summary_rows)
        raw_repeat_tvd = {
            int(row["repeat_index"]): float(row["tvd_to_ideal"])
            for _, row in seed_summary.loc[
                seed_summary["scenario"] == "raw_haiqu_execution"
            ].iterrows()
        }
        raw_repeat_success = {
            int(row["repeat_index"]): float(row["success_probability"])
            for _, row in seed_summary.loc[
                seed_summary["scenario"] == "raw_haiqu_execution"
            ].iterrows()
        }
        for summary_index, summary_row in summary.iterrows():
            scenario = str(summary_row["scenario"])
            scenario_repeats = seed_summary.loc[
                seed_summary["scenario"] == scenario
            ]
            tvd_improvements = [
                raw_repeat_tvd[int(row["repeat_index"])]
                - float(row["tvd_to_ideal"])
                for _, row in scenario_repeats.iterrows()
                if int(row["repeat_index"]) in raw_repeat_tvd
            ]
            success_gains = [
                float(row["success_probability"])
                - raw_repeat_success[int(row["repeat_index"])]
                for _, row in scenario_repeats.iterrows()
                if int(row["repeat_index"]) in raw_repeat_success
            ]
            (
                paired_tvd_mean,
                paired_tvd_low,
                paired_tvd_high,
            ) = _mean_ci95(tvd_improvements)
            (
                paired_success_mean,
                paired_success_low,
                paired_success_high,
            ) = _mean_ci95(success_gains)
            summary.loc[
                summary_index,
                "paired_tvd_improvement_vs_raw",
            ] = paired_tvd_mean if np.isfinite(paired_tvd_mean) else None
            summary.loc[
                summary_index,
                "paired_tvd_improvement_ci95_low",
            ] = paired_tvd_low
            summary.loc[
                summary_index,
                "paired_tvd_improvement_ci95_high",
            ] = paired_tvd_high
            summary.loc[
                summary_index,
                "paired_success_gain_vs_raw",
            ] = paired_success_mean if np.isfinite(paired_success_mean) else None
            summary.loc[
                summary_index,
                "paired_success_gain_ci95_low",
            ] = paired_success_low
            summary.loc[
                summary_index,
                "paired_success_gain_ci95_high",
            ] = paired_success_high
            summary.loc[summary_index, "ci_method"] = (
                "student_t_over_independent_repeats"
                if len(tvd_improvements) >= 2
                else "not_available_fewer_than_two_repeats"
            )
            summary.loc[
                summary_index,
                "uncertainty_basis",
            ] = "paired_repeat_means_over_fixed_edge_target_depth_suite"
        raw_value = summary.loc[
            summary["scenario"] == "raw_haiqu_execution",
            "tvd_to_ideal",
        ].iloc[0]
        corrected_value = summary.loc[
            summary["scenario"]
            == "learned_readout_plus_generator_correction",
            "tvd_to_ideal",
        ].iloc[0]
        improvement = (
            float(raw_value - corrected_value)
            if pd.notna(raw_value) and pd.notna(corrected_value)
            else None
        )
        response_payload = {
            "schema_version": "0.5",
            "algorithm": "grover_2q_exhaustive",
            "targets": list(GROVER_TARGETS),
            "two_qubit_depths": list(
                self.config.benchmark_two_qubit_depths
            ),
            "depth_sweep_method": "cx_cx_identity_folding",
            "uses_benchmark_counts_for_response_fit": False,
            "uses_evaluation_counts_for_stack_selection": False,
            "transpilation_optimization_level": self.config.optimization_level,
            "response_regularization": self.config.response_regularization,
            "validation": validation_summary,
            "evaluation": {
                "repeats": self.config.evaluation_repeats,
                "seeds": evaluation_seeds,
                "held_out_from_validation": True,
            },
            "correction_stack": {
                "readout": (
                    "independent 1Q/2Q assignment calibration; applied to raw "
                    "diagnostics before the clean-feature Ridge model and to "
                    "Grover distributions before generator inversion"
                ),
                "alpha": (
                    "propagated after every modeled gate and included in the "
                    "regularized algorithm-response inverse"
                ),
                "gamma": "included in the regularized algorithm-response inverse",
                "kappa_down": (
                    "non-unital component included in the same response inverse"
                ),
                "selection": (
                    "the generator inverse is enabled independently per "
                    "physical edge on validation data; rejected edges retain "
                    "the readout-only result"
                ),
                "haiqu": (
                    "optional SDK mitigation layer; local readout inversion is "
                    "not duplicated after Haiqu advanced mitigation"
                ),
                "added_physical_correction_gates": 0,
            },
            "readout_calibration": readout_calibration.to_dict(),
            "semantics": (
                "validated algorithm-level error cancellation and mitigation; "
                "not fault-tolerant QEC"
            ),
            "models": {
                edge_key: model.to_dict()
                for edge_key, model in response_models.items()
            },
            "correction_ready": correction_ready,
            "correction_enabled": correction_enabled,
            "generator_enabled_edges": enabled_edges,
            "readout_only_fallback_edges": fallback_edges,
            "generator_enabled_edge_depths": enabled_edge_depths,
            "readout_only_fallback_edge_depths": fallback_edge_depths,
            "correction_unavailable_reason": correction_unavailable_reason,
        }
        benchmark_analytics = (
            pd.concat(analytics_frames, ignore_index=True)
            if analytics_frames
            else pd.DataFrame()
        )
        validation_frame = pd.DataFrame(validation_rows)
        self.session.log_object(
            summary,
            name="Grover v0.5 repeated benchmark summary",
            group="final",
        )
        self.session.log_object(
            validation_frame,
            name="Grover correction validation split",
            group="final",
        )
        self.session.log_object(
            details,
            name="Grover held-out evaluation instances",
            group="final",
        )
        self.session.log_object(
            response_payload,
            name="readout and generator response models",
            group="final",
        )
        return (
            summary,
            details,
            validation_frame,
            seed_summary,
            benchmark_analytics,
            response_payload,
            improvement,
        )

    def _deployment_branch(
        self,
        qnn_analytics: pd.DataFrame,
    ) -> pd.DataFrame:
        """Report the circuit that actually produced the QNN reconstruction."""

        if qnn_analytics.empty:
            return pd.DataFrame(
                [
                    {
                        "model_kind": "ridge",
                        "encoding": "none_classical_baseline",
                        "status": "not_requested_model_mode_ridge",
                    }
                ]
            )
        deployment = qnn_analytics.copy()
        deployment["stage"] = "trained_qnn_inference"
        deployment["encoding"] = "trained_angle_data_reuploading"
        deployment["trained_parameters"] = True
        deployment["classical_readout"] = "saved_ridge_head"
        return deployment

    def _amplitude_model_available(self) -> bool:
        """Read explicit encoding metadata instead of grepping prose."""

        for path in self.config.model_root.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            families = payload.get("families", {})
            if not isinstance(families, dict):
                continue
            for family in families.values():
                if not isinstance(family, dict):
                    continue
                architectures = family.get("architectures", {})
                if not isinstance(architectures, dict):
                    continue
                for architecture in architectures.values():
                    if not isinstance(architecture, dict):
                        continue
                    encoding = str(architecture.get("encoding", "")).lower()
                    if encoding in {"amplitude", "amplitude_encoding"}:
                        return True
        return False

    @staticmethod
    def _plot_benchmark(frame: pd.DataFrame, path: Path) -> Path:
        executed = frame[
            frame["status"].fillna("").str.startswith("executed")
            & frame["scenario"].isin(_PRESENTATION_SCENARIOS)
        ].copy()
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
        display_names = {
            "ideal_simulation": "Ideal",
            "raw_haiqu_execution": "Raw Haiqu",
            "calibrated_readout_mitigation_only": "Calibrated\nreadout",
            "learned_readout_plus_generator_correction": (
                "Learned correction\n(validated)"
            ),
            "learned_full_correction_plus_haiqu_mitigation": (
                "Learned +\nHaiqu"
            ),
        }
        labels = []
        for _, row in executed.iterrows():
            label = display_names.get(row["scenario"], row["scenario"])
            if (
                row["scenario"]
                == "learned_readout_plus_generator_correction"
                and "fallback_no_generator" in str(row["status"])
            ):
                label = "Learned not applied\n(readout fallback)"
            if row["status"] == "executed_local_fallback_not_haiqu":
                label = label.replace("Haiqu", "local")
            labels.append(label)
        has_ci = bool(
            not executed.empty
            and executed["repeat_count"].ge(2).all()
            and executed["tvd_ci95_low"].notna().all()
            and executed["success_probability_ci95_low"].notna().all()
        )
        tvd_error = (
            executed["tvd_to_ideal"] - executed["tvd_ci95_low"]
            if has_ci
            else None
        )
        success_error = (
            executed["success_probability"]
            - executed["success_probability_ci95_low"]
            if has_ci
            else None
        )
        tvd_bars = axes[0].bar(
            labels,
            executed["tvd_to_ideal"],
            yerr=tvd_error,
            capsize=3,
            color="#ef8354",
        )
        axes[0].set_ylabel("Mean total variation distance")
        axes[0].set_title("Lower is better")
        success_bars = axes[1].bar(
            labels,
            executed["success_probability"],
            yerr=success_error,
            capsize=3,
            color="#2d6a9f",
        )
        axes[1].set_ylim(0.0, 1.05)
        axes[1].set_ylabel("Mean target success probability")
        axes[1].set_title("Higher is better")
        axes[0].bar_label(tvd_bars, fmt="%.4f", padding=4, fontsize=8)
        axes[1].bar_label(success_bars, fmt="%.4f", padding=4, fontsize=8)
        for axis in axes:
            axis.tick_params(axis="x", labelrotation=0, labelsize=8)
            axis.grid(axis="y", alpha=0.25)
        uncertainty_caption = (
            "Error bars: 95% Student-t CI over independent held-out repeats"
            if has_ci
            else "Single held-out repeat: confidence intervals are not available"
        )
        fig.suptitle(
            "Q-ErrorID 2Q Grover depth sweep "
            "(4 targets × 3 edges × configured depths)\n"
            f"{uncertainty_caption}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _plot_depth_sweep(frame: pd.DataFrame, path: Path) -> Path:
        fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        display_names = {
            "raw_haiqu_execution": "Raw",
            "calibrated_readout_mitigation_only": "Readout-only",
            "learned_readout_plus_generator_correction": (
                "Learned correction (validated)"
            ),
            "learned_full_correction_plus_haiqu_mitigation": (
                "Learned + Haiqu"
            ),
        }
        if "scenario" not in frame:
            frame = pd.DataFrame(
                columns=[
                    "scenario",
                    "requested_two_qubit_depth",
                    "repeat_count",
                    "tvd_to_ideal",
                    "tvd_ci95_low",
                    "learned_generator_applied_fraction",
                ]
            )
        for scenario in _PRESENTATION_SCENARIOS:
            group = frame.loc[frame["scenario"] == scenario].sort_values(
                "requested_two_qubit_depth"
            )
            if group.empty:
                continue
            label = display_names[scenario]
            applied_fraction = float(
                group["learned_generator_applied_fraction"].max()
            )
            if (
                scenario
                == "learned_readout_plus_generator_correction"
                and applied_fraction == 0.0
            ):
                label = "Learned not applied (readout fallback)"
            elif (
                scenario
                == "learned_full_correction_plus_haiqu_mitigation"
                and applied_fraction == 0.0
            ):
                label = "Learned not applied + Haiqu"
            has_ci = bool(
                group["repeat_count"].ge(2).all()
                and group["tvd_ci95_low"].notna().all()
            )
            yerr = (
                group["tvd_to_ideal"] - group["tvd_ci95_low"]
                if has_ci
                else None
            )
            axis.errorbar(
                group["requested_two_qubit_depth"],
                group["tvd_to_ideal"],
                yerr=yerr,
                marker="o",
                capsize=3,
                label=label,
            )
        axis.set_xlabel("Requested two-qubit depth")
        axis.set_ylabel("Mean total variation distance")
        axis.set_title(
            "Ideal-preserving CX-folding depth sweep\n"
            "Confidence intervals are shown only with at least two repeats"
        )
        axis.set_xticks(
            sorted(frame["requested_two_qubit_depth"].unique())
            if not frame.empty
            else []
        )
        axis.grid(alpha=0.25)
        if not frame.empty:
            axis.legend()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    @staticmethod
    def _plot_mitigation(frame: pd.DataFrame, path: Path) -> Path:
        executed = frame[frame["status"] == "executed"].copy()
        rows = []
        for mode, group in executed.groupby("mode", sort=False):
            alpha = [
                np.linalg.norm(list(json.loads(value).values()))
                for value in group["predicted_alpha"]
            ]
            gamma = [
                np.linalg.norm(list(json.loads(value).values()))
                for value in group["predicted_gamma"]
            ]
            rows.append((mode, np.mean(alpha), np.mean(gamma)))
        fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
        if rows:
            x = np.arange(len(rows))
            axis.bar(x - 0.18, [row[1] for row in rows], 0.36, label="coherent")
            axis.bar(x + 0.18, [row[2] for row in rows], 0.36, label="stochastic")
            axis.set_xticks(x, [row[0].replace("_", "\n") for row in rows])
        axis.set_ylabel("Mean reconstructed magnitude")
        axis.set_title("Raw and Haiqu mitigation modes")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return path

    def _log_saved_figure(self, path: Path, *, name: str, group: str) -> None:
        if not self.session.cloud_enabled:
            return
        image = mpimg.imread(path)
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.imshow(image)
        axis.set_axis_off()
        self.session.log_object(figure, name=name, group=group)
        plt.close(figure)

    def run(self, *, include_benchmark: bool = True) -> PipelineReport:
        self._progress("Authenticating runtime and selecting the device")
        self.session.authenticate()
        selected = self.session.select_device(self.config.device)
        qubits, edges = self.session.subgraph()
        self._progress(
            f"Selected {selected.id} on qubits {qubits}; building diagnostics"
        )
        circuits, edge_qubits = self._build_diagnostics(qubits, edges)
        readout_circuits = build_readout_calibration_circuits(
            qubits,
            edges,
            width=len(qubits),
        )
        all_characterization_circuits = [*circuits, *readout_circuits]

        diagnostics = diagnostic_table(circuits)
        readout_circuit_table = pd.DataFrame(
            [
                {
                    "circuit_name": circuit.name,
                    **(circuit.metadata or {}),
                }
                for circuit in readout_circuits
            ]
        )
        original_analytics = pd.DataFrame(
            [
                {
                    **local_circuit_analytics(
                        circuit,
                        stage="original",
                        device_id=selected.id,
                    ),
                    "circuit_kind": (
                        "readout_calibration"
                        if (circuit.metadata or {}).get("calibration_type")
                        == "readout_assignment"
                        else "diagnostic"
                    ),
                }
                for circuit in all_characterization_circuits
            ]
        )
        if self.session.cloud_enabled:
            self._progress(
                "Submitting raw diagnostics and readout calibration to Haiqu"
            )
            logged = self.session.log_circuits(
                all_characterization_circuits,
                group="diagnostics",
            )
            (
                all_execution_circuits,
                transpiled_analytics,
            ) = self.session.transpile_cloud(
                logged,
            )
            all_raw_results, raw_job = self.session.run_cloud(
                all_execution_circuits,
                mode=MitigationMode.RAW,
                group="diagnostics",
                job_name="Q-ErrorID raw diagnostics and readout calibration",
            )
        else:
            self._progress(
                "Running raw diagnostics and readout calibration locally"
            )
            (
                all_execution_circuits,
                all_raw_results,
                transpiled_analytics,
            ) = (
                self.session.local_transpile_and_run(
                    all_characterization_circuits,
                    physical_qubits=qubits,
                    tree_edges=edges,
                )
            )
            raw_job = None
        diagnostic_count = len(circuits)
        execution_circuits = all_execution_circuits[:diagnostic_count]
        raw_results = all_raw_results[:diagnostic_count]
        readout_results = all_raw_results[diagnostic_count:]
        transpiled_analytics = transpiled_analytics.copy()
        transpiled_analytics["circuit_kind"] = [
            "diagnostic" if index < diagnostic_count else "readout_calibration"
            for index in range(len(transpiled_analytics))
        ]
        readout_calibration = ReadoutCalibrationBundle.from_results(
            readout_circuits,
            readout_results,
            regularization=self.config.readout_regularization,
            expected_shots=self.config.shots,
        )
        if not readout_calibration.validation_passed:
            raise RuntimeError(
                "At least one independently measured readout assignment matrix "
                "failed validation"
            )
        self._progress(
            "Readout calibration passed; reconstructing physical generators"
        )
        ideal_results = _ideal_counts(
            circuits,
            shots=self.config.shots,
            seed=self.config.seed,
        )
        raw_batch = results_to_features(circuits, raw_results, mode="raw")
        calibrated_batch = results_to_features(
            circuits,
            raw_results,
            mode="readout_mitigated",
            readout_calibration=readout_calibration,
        )
        ideal_batch = results_to_features(circuits, ideal_results, mode="ideal")
        model_mode = self.config.model_mode
        primary_model_kind = "qnn" if model_mode == "qnn" else "ridge"
        raw_audit_estimates, raw_audit_rows = self._estimate_batch(
            raw_batch,
            ideal_batch,
            model_kind=primary_model_kind,
            inference_backend=(
                "numpy_exact_statevector_postprocessor"
                if primary_model_kind == "qnn"
                else "classical_cpu"
            ),
            feature_regime="raw_shot",
            feature_stage="raw_unmitigated_audit",
        )
        ridge_estimates: dict[str, ChannelEstimate] = {}
        ridge_rows: list[dict[str, Any]] = []
        if model_mode in {"ridge", "compare"}:
            ridge_estimates, ridge_rows = self._estimate_batch(
                calibrated_batch,
                ideal_batch,
                model_kind="ridge",
                inference_backend="classical_cpu",
                feature_regime="readout_corrected",
                feature_stage="readout_mitigated",
            )

        qnn_estimates: dict[str, ChannelEstimate] = {}
        qnn_rows: list[dict[str, Any]] = []
        qnn_analytics = pd.DataFrame()
        if model_mode in {"qnn", "compare"}:
            self._progress("Executing seven trained QNN inference circuits")
            qnn_estimates, qnn_rows, qnn_analytics = self._execute_qnn_batch(
                calibrated_batch,
                qubits,
                edges,
                readout_calibration,
            )

        raw_estimates = (
            qnn_estimates if primary_model_kind == "qnn" else ridge_estimates
        )
        if set(raw_estimates) != set(calibrated_batch.features):
            raise RuntimeError(
                f"{primary_model_kind} inference did not reconstruct every channel"
            )
        reconstructed_rows = [*ridge_rows, *qnn_rows]
        for row in reconstructed_rows:
            row["is_primary"] = row["model_kind"] == primary_model_kind

        single_estimates = {
            physical: raw_estimates[f"q{physical}"] for physical in qubits
        }
        two_estimates = {edge: raw_estimates[edge] for edge in edge_qubits}
        atlas = DeviceErrorAtlas.from_estimates(
            device_id=selected.id,
            physical_qubits=qubits,
            single_qubit_estimates=single_estimates,
            two_qubit_estimates=two_estimates,
            edge_qubits=edge_qubits,
            calibration_timestamp=selected.calibration_timestamp,
            metadata={
                "track": "empirical_device_characterization",
                "ground_truth_known": False,
                "raw_counts_preserved": True,
                "generator_fit_input": "independently_readout_calibrated_features",
                "readout_calibration_validation_passed": (
                    readout_calibration.validation_passed
                ),
                "execution_source": selected.source,
            },
        )

        mitigation = self._mitigation_modes(
            execution_circuits,
            circuits,
            ideal_batch,
            raw_audit_estimates,
            raw_audit_rows,
            qnn_rows if primary_model_kind == "qnn" else ridge_rows,
            primary_model_kind,
            raw_job,
        )
        synthetic = self._synthetic_ground_truth_track()
        deployment = self._deployment_branch(qnn_analytics)
        analytics = pd.concat(
            [original_analytics, transpiled_analytics, qnn_analytics],
            ignore_index=True,
        )

        diagnostic_path = _save_dataframe(
            diagnostics,
            self.result_dir / "diagnostic_circuits.csv",
        )
        readout_circuit_path = _save_dataframe(
            readout_circuit_table,
            self.result_dir / "readout_calibration_circuits.csv",
        )
        readout_table_path = _save_dataframe(
            readout_calibration.dataframe(),
            self.result_dir / "readout_calibration.csv",
        )
        readout_json_path = self.artifact_dir / "readout_calibration.json"
        readout_json_path.write_text(
            json.dumps(
                readout_calibration.to_dict(),
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        analytics_path = _save_dataframe(
            analytics,
            self.result_dir / "circuit_analytics.csv",
        )
        feature_path = _save_dataframe(
            pd.concat(
                [ideal_batch.table, raw_batch.table, calibrated_batch.table],
                ignore_index=True,
            ),
            self.result_dir / "feature_matrices.csv",
        )
        reconstruction_path = _save_dataframe(
            pd.DataFrame(reconstructed_rows),
            self.result_dir / "reconstructed_channels.csv",
        )
        mitigation_path = _save_dataframe(
            mitigation,
            self.result_dir / "mitigation_comparison.csv",
        )
        synthetic_path = _save_dataframe(
            synthetic,
            self.result_dir / "synthetic_ground_truth.csv",
        )
        deployment_path = _save_dataframe(
            deployment,
            self.result_dir / "model_deployment.csv",
        )
        atlas_json_path = atlas.save_json(self.artifact_dir / "device_error_atlas.json")
        atlas_table_path = _save_dataframe(
            atlas.dataframe(),
            self.result_dir / "device_error_atlas.csv",
        )
        atlas_plot_path = atlas.plot(self.result_dir / "device_error_atlas.png")
        mitigation_plot_path = self._plot_mitigation(
            mitigation,
            self.result_dir / "mitigation_comparison.png",
        )

        benchmark_path = self.result_dir / "final_benchmark.csv"
        benchmark_plot_path = self.result_dir / "final_benchmark.png"
        presentation_benchmark_path = (
            self.result_dir / "presentation_benchmark.csv"
        )
        depth_sweep_path = self.result_dir / "depth_sweep_benchmark.csv"
        depth_sweep_plot_path = self.result_dir / "depth_sweep_benchmark.png"
        rejected_ablation_path = (
            self.result_dir / "rejected_generator_ablation.csv"
        )
        execution_audit_path = self.artifact_dir / "execution_audit.json"
        benchmark_detail_path = (
            self.result_dir / "algorithm_benchmark_details.csv"
        )
        validation_path = self.result_dir / "correction_validation.csv"
        seed_summary_path = self.result_dir / "benchmark_seed_summary.csv"
        response_model_path = (
            self.artifact_dir / "algorithm_response_models.json"
        )
        improvement = None
        if include_benchmark:
            self._progress(
                "Running reserved validation repeats and held-out Grover evaluation"
            )
            (
                benchmark,
                benchmark_details,
                validation_frame,
                seed_summary,
                benchmark_analytics,
                response_payload,
                improvement,
            ) = self._benchmark(
                atlas,
                qubits,
                edges,
                readout_calibration,
            )
            _save_dataframe(benchmark, benchmark_path)
            _save_dataframe(benchmark_details, benchmark_detail_path)
            _save_dataframe(validation_frame, validation_path)
            _save_dataframe(seed_summary, seed_summary_path)
            presentation = benchmark.loc[
                benchmark["scenario"].isin(_PRESENTATION_SCENARIOS)
            ].copy()
            presentation_names = {
                "raw_haiqu_execution": "Raw",
                "calibrated_readout_mitigation_only": "Readout-only",
                "learned_readout_plus_generator_correction": (
                    "Learned correction (validation-gated)"
                ),
                "learned_full_correction_plus_haiqu_mitigation": (
                    "Learned + Haiqu"
                ),
            }
            presentation["presentation_label"] = presentation[
                "scenario"
            ].map(presentation_names)
            presentation["qnn_executed"] = bool(qnn_rows)
            presentation["qnn_circuit_count"] = len(qnn_rows)
            applied_rows = benchmark_details.loc[
                benchmark_details["scenario"]
                == "learned_readout_plus_generator_correction"
            ]
            learned_applied_fraction = (
                float(
                    applied_rows["generator_correction_applied"]
                    .fillna(False)
                    .astype(bool)
                    .mean()
                )
                if not applied_rows.empty
                else 0.0
            )
            presentation["learned_generator_applied_fraction"] = np.nan
            presentation["learned_generator_contribution_status"] = (
                "not_applicable"
            )
            learned_mask = presentation["scenario"].isin(
                (
                    "learned_readout_plus_generator_correction",
                    "learned_full_correction_plus_haiqu_mitigation",
                )
            )
            presentation.loc[
                learned_mask,
                "learned_generator_applied_fraction",
            ] = learned_applied_fraction
            presentation.loc[
                learned_mask,
                "learned_generator_contribution_status",
            ] = (
                "applied_on_validation_selected_edge_depths"
                if learned_applied_fraction > 0.0
                else "not_applied_readout_only_fallback"
            )
            _save_dataframe(presentation, presentation_benchmark_path)

            depth_sweep = _depth_sweep_summary(benchmark_details)
            _save_dataframe(depth_sweep, depth_sweep_path)
            self._plot_depth_sweep(depth_sweep, depth_sweep_plot_path)
            rejected_ablation = benchmark_details.loc[
                benchmark_details["scenario"]
                == "learned_generator_correction_only"
            ].copy()
            if not rejected_ablation.empty:
                rejected_ablation["reporting_role"] = (
                    "rejected_diagnostic_ablation_not_a_claimed_result"
                )
            _save_dataframe(rejected_ablation, rejected_ablation_path)

            executed_repeats = int(
                benchmark.loc[
                    benchmark["scenario"] == "raw_haiqu_execution",
                    "repeat_count",
                ].iloc[0]
            )
            execution_audit = {
                "schema_version": "0.6",
                "execution_mode": (
                    "haiqu_cloud"
                    if self.session.cloud_enabled
                    else "local_fallback"
                ),
                "device_id": selected.id,
                "shots": self.config.shots,
                "validation_repeats": self.config.validation_repeats,
                "evaluation_repeats": self.config.evaluation_repeats,
                "two_qubit_depths": list(
                    self.config.benchmark_two_qubit_depths
                ),
                "depth_sweep_method": "cx_cx_identity_folding",
                "confidence_intervals_available": executed_repeats >= 2,
                "confidence_interval_method": (
                    "student_t_over_independent_repeat_means"
                    if executed_repeats >= 2
                    else "not_available_fewer_than_two_repeats"
                ),
                "paired_improvements_reported": True,
                "model_mode_requested": model_mode,
                "ridge_executed": bool(ridge_rows),
                "qnn_executed": bool(qnn_rows),
                "qnn_circuit_count": len(qnn_rows),
                "qnn_expected_circuit_count": 7,
                "generator_only_ablation_in_main_figure": False,
                "generator_only_ablation_role": (
                    "rejected diagnostic with negativity and simplex "
                    "projection audit"
                ),
                "learned_generator_applied_fraction": (
                    learned_applied_fraction
                ),
            }
            execution_audit_path.write_text(
                json.dumps(execution_audit, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            response_model_path.write_text(
                json.dumps(
                    response_payload,
                    indent=2,
                    sort_keys=True,
                    default=str,
                ),
                encoding="utf-8",
            )
            if not benchmark_analytics.empty:
                analytics = pd.concat(
                    [analytics, benchmark_analytics],
                    ignore_index=True,
                )
                _save_dataframe(analytics, analytics_path)
            self._plot_benchmark(benchmark, benchmark_plot_path)
            self._progress("Grover validation and evaluation completed")
        else:
            skipped = pd.DataFrame(
                [{"status": "skipped_by_run_haiqu_pipeline"}]
            )
            _save_dataframe(
                skipped,
                benchmark_path,
            )
            _save_dataframe(
                pd.DataFrame([{"status": "skipped_by_run_haiqu_pipeline"}]),
                benchmark_detail_path,
            )
            _save_dataframe(
                pd.DataFrame([{"status": "skipped_by_run_haiqu_pipeline"}]),
                validation_path,
            )
            _save_dataframe(
                pd.DataFrame([{"status": "skipped_by_run_haiqu_pipeline"}]),
                seed_summary_path,
            )
            _save_dataframe(skipped, presentation_benchmark_path)
            _save_dataframe(skipped, depth_sweep_path)
            _save_dataframe(skipped, rejected_ablation_path)
            self._plot_depth_sweep(pd.DataFrame(), depth_sweep_plot_path)
            execution_audit_path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.6",
                        "benchmark_status": (
                            "skipped_by_run_haiqu_pipeline"
                        ),
                        "model_mode_requested": model_mode,
                        "ridge_executed": bool(ridge_rows),
                        "qnn_executed": bool(qnn_rows),
                        "qnn_circuit_count": len(qnn_rows),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            response_model_path.write_text(
                json.dumps(
                    {
                        "status": "skipped_by_run_haiqu_pipeline",
                        "algorithm": "grover_2q_exhaustive",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            self._plot_benchmark(
                pd.DataFrame(
                    columns=[
                        "scenario",
                        "status",
                        "tvd_to_ideal",
                        "success_probability",
                    ]
                ),
                benchmark_plot_path,
            )

        self.session.log_object(
            diagnostics,
            name="diagnostic circuit table",
            group="diagnostics",
        )
        self.session.log_object(
            readout_calibration.dataframe(),
            name="independent readout assignment calibration",
            group="diagnostics",
        )
        self.session.log_object(
            analytics,
            name="circuit analytics table",
            group="diagnostics",
        )
        self.session.log_object(
            raw_batch.table,
            name="raw counts and feature table",
            group="diagnostics",
        )
        self.session.log_object(
            atlas.dataframe(),
            name="device error atlas table",
            group="model",
        )
        self.session.log_object(
            synthetic,
            name="synthetic true-versus-predicted",
            group="model",
        )
        self.session.log_object(
            {
                "model_root": self.config.model_root.as_posix(),
                "model_files": [path.name for path in self.config.model_root.glob("*")],
                "dataset_root": self.config.data_root.as_posix(),
                "dataset_files": [
                    path.name for path in self.config.data_root.glob("*")
                ],
            },
            name="model and dataset manifests",
            group="model",
        )
        self.session.log_object(
            mitigation,
            name="mitigation mode comparison",
            group="mitigation",
        )
        self._log_saved_figure(
            atlas_plot_path,
            name="device error atlas",
            group="model",
        )
        self._log_saved_figure(
            mitigation_plot_path,
            name="mitigation comparison",
            group="mitigation",
        )
        self._log_saved_figure(
            benchmark_plot_path,
            name="final benchmark",
            group="final",
        )

        hybrid_schema = None
        hybrid_schema_status = "not_validated_sdk_unavailable"
        if self.session.sdk_version != "not-installed":
            hybrid_schema = self.session.hybrid_program(
                MitigationMode.ADVANCED
            ).model_dump(mode="json")
            hybrid_schema_status = "validated"

        manifest = self.session.manifest()
        manifest.update(
            {
                "selected_subgraph": list(qubits),
                "selected_edges": [list(edge) for edge in edges],
                "tracks": {
                    "synthetic_ground_truth": _portable_path(synthetic_path, self.root),
                    "empirical_device_characterization": _portable_path(
                        mitigation_path, self.root
                    ),
                    "algorithm_benchmark": _portable_path(
                        benchmark_path,
                        self.root,
                    ),
                    "algorithm_benchmark_instances": _portable_path(
                        benchmark_detail_path,
                        self.root,
                    ),
                    "correction_validation": _portable_path(
                        validation_path,
                        self.root,
                    ),
                    "benchmark_seed_summary": _portable_path(
                        seed_summary_path,
                        self.root,
                    ),
                    "presentation_benchmark": _portable_path(
                        presentation_benchmark_path,
                        self.root,
                    ),
                    "depth_sweep_benchmark": _portable_path(
                        depth_sweep_path,
                        self.root,
                    ),
                    "rejected_generator_ablation": _portable_path(
                        rejected_ablation_path,
                        self.root,
                    ),
                    "learned_response_models": _portable_path(
                        response_model_path,
                        self.root,
                    ),
                    "readout_calibration": _portable_path(
                        readout_json_path,
                        self.root,
                    ),
                },
                "raw_characterization_mode": {
                    "use_mitigation": False,
                    "raw_counts_preserved": True,
                    "readout_calibration_circuits": len(readout_circuits),
                    "generator_reconstruction_input": (
                        "raw counts corrected only by independently measured "
                        "assignment matrices"
                    ),
                },
                "model_inference": {
                    "requested_mode": model_mode,
                    "primary_model_kind": primary_model_kind,
                    "ridge_executed": bool(ridge_rows),
                    "trained_qnn_circuits_executed": bool(qnn_rows),
                    "qnn_execution_backend": (
                        qnn_rows[0]["inference_backend"] if qnn_rows else None
                    ),
                    "qnn_classical_readout": (
                        "saved_ridge_head" if qnn_rows else None
                    ),
                },
                "hybrid_flow": {
                    "program_schema": hybrid_schema,
                    "schema_status": hybrid_schema_status,
                    "custom_model_inference_layer_supported": False,
                    "integration": (
                        "trained_qnn_circuits_run_via_Haiqu_sdk"
                        if self.session.cloud_enabled and qnn_rows
                        else "trained_qnn_circuits_run_in_explicit_local_fallback"
                        if qnn_rows
                        else "classical_ridge_only"
                    ),
                },
                "correction_improvement_tvd": improvement,
                "execution_audit": (
                    _portable_path(execution_audit_path, self.root)
                    if include_benchmark
                    else None
                ),
                "readout_calibration": {
                    "validation_passed": readout_calibration.validation_passed,
                    "single_qubit_matrices": len(
                        readout_calibration.single_qubit
                    ),
                    "two_qubit_matrices": len(readout_calibration.two_qubit),
                    "regularization": readout_calibration.regularization,
                    "artifact": _portable_path(readout_json_path, self.root),
                },
                "algorithm_correction": {
                    "algorithm": "two_qubit_grover_search",
                    "instances": (
                        len(edges)
                        * len(GROVER_TARGETS)
                        * len(self.config.benchmark_two_qubit_depths)
                    ),
                    "targets_per_edge": list(GROVER_TARGETS),
                    "two_qubit_depths": list(
                        self.config.benchmark_two_qubit_depths
                    ),
                    "depth_sweep_method": "cx_cx_identity_folding",
                    "coherent_alpha": (
                        "propagated through every modeled 1Q/CX error location "
                        "and included in the full algorithm-response inverse"
                    ),
                    "stochastic_gamma": (
                        "regularized inverse of the atlas-predicted algorithm "
                        "response matrix"
                    ),
                    "amplitude_damping_kappa": (
                        "non-unital component included in the same response inverse"
                    ),
                    "uses_benchmark_counts_to_fit_inverse": False,
                    "added_physical_correction_gates": 0,
                    "semantics": (
                        "error cancellation and mitigation, not "
                        "fault-tolerant quantum error correction"
                    ),
                },
                "limitations": [
                    "Backend predictions are not presented as exact ground truth.",
                    "Dynamical decoupling is not claimed to reverse amplitude damping.",
                    (
                        "The regularized response inverse is non-CPTP and can amplify "
                        "finite-shot uncertainty."
                    ),
                    (
                        "Correction quality is limited by mismatch between hardware "
                        "noise and the reconstructed generator ansatz."
                    ),
                    (
                        "On devices larger than the four-qubit fake_fez target, "
                        "the cloud transpiler's final physical layout must be "
                        "verified before interpreting edge-specific correction."
                    ),
                ],
            }
        )
        manifest_path = self.artifact_dir / "haiqu_experiments.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self.session.log_object(
            manifest,
            name="Haiqu experiment manifest",
            group="final",
        )
        self._progress("Saved v0.6 tables, audits, figures, and manifest")

        artifacts = {
            "haiqu_manifest": str(manifest_path),
            "atlas_json": str(atlas_json_path),
            "diagnostic_table": str(diagnostic_path),
            "readout_calibration_circuits": str(readout_circuit_path),
            "readout_calibration_table": str(readout_table_path),
            "readout_calibration_json": str(readout_json_path),
            "circuit_analytics": str(analytics_path),
            "feature_matrices": str(feature_path),
            "reconstructed_channels": str(reconstruction_path),
            "mitigation_comparison": str(mitigation_path),
            "synthetic_ground_truth": str(synthetic_path),
            "model_deployment": str(deployment_path),
            "atlas_table": str(atlas_table_path),
            "atlas_plot": str(atlas_plot_path),
            "mitigation_plot": str(mitigation_plot_path),
            "final_benchmark": str(benchmark_path),
            "algorithm_benchmark_details": str(benchmark_detail_path),
            "correction_validation": str(validation_path),
            "benchmark_seed_summary": str(seed_summary_path),
            "algorithm_response_models": str(response_model_path),
            "final_benchmark_plot": str(benchmark_plot_path),
            "presentation_benchmark": str(presentation_benchmark_path),
            "depth_sweep_benchmark": str(depth_sweep_path),
            "depth_sweep_plot": str(depth_sweep_plot_path),
            "rejected_generator_ablation": str(rejected_ablation_path),
            "execution_audit": str(execution_audit_path),
        }
        return PipelineReport(
            execution_mode=(
                "haiqu_cloud" if self.session.cloud_enabled else "local_fallback"
            ),
            device_id=selected.id,
            physical_qubits=qubits,
            experiments=self.session.experiments,
            artifacts=artifacts,
            correction_improvement_tvd=improvement,
            unsupported_features=(
                []
                if self.session.cloud_enabled
                else [
                    "Haiqu experiment tracking",
                    "Haiqu mitigation modes",
                    "Haiqu vector loading",
                    "Haiqu hybrid flow execution",
                ]
            ),
        )
