"""Haiqu cloud adapter and honest local validation runtime."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .circuits import local_circuit_analytics
from .config import EXPERIMENT_GROUPS, ExecutionConfig, MitigationMode


def choose_connected_subgraph(
    coupling_edges: Sequence[Sequence[int]],
    *,
    size: int = 4,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Choose a deterministic connected subgraph and a three-edge spanning tree."""

    adjacency: dict[int, set[int]] = {}
    for raw_left, raw_right in coupling_edges:
        left, right = int(raw_left), int(raw_right)
        if left == right:
            continue
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    for start in sorted(adjacency):
        selected = [start]
        parents: dict[int, int] = {}
        cursor = 0
        while cursor < len(selected) and len(selected) < size:
            current = selected[cursor]
            cursor += 1
            for neighbor in sorted(adjacency[current]):
                if neighbor in selected:
                    continue
                selected.append(neighbor)
                parents[neighbor] = current
                if len(selected) == size:
                    break
        if len(selected) == size:
            tree = tuple((parents[node], node) for node in selected[1:])
            return tuple(selected), tree
    raise ValueError(f"No connected {size}-qubit subgraph is available")


def _linear_chain_coupling(num_qubits: int) -> list[list[int]]:
    """Return a bidirectional nearest-neighbour coupling map of the given size."""

    edges: list[list[int]] = []
    for qubit in range(num_qubits - 1):
        edges.append([qubit, qubit + 1])
        edges.append([qubit + 1, qubit])
    return edges


def _fake_backend(device_id: str, seed: int, num_qubits: int = 4):
    """Return an IBM fake backend, or a dependency-light generic substitute."""

    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError:
        fake_provider = None

    class_name = "".join(part.capitalize() for part in device_id.split("_"))
    backend_type = (
        getattr(fake_provider, class_name, None) if fake_provider is not None else None
    )
    if backend_type is not None:
        return backend_type()
    if device_id != "fake_fez":
        raise ValueError(f"Local fake backend is unavailable: {device_id}")

    # qiskit-ibm-runtime is intentionally not required for the honest local
    # fallback. GenericBackendV2 supplies a deterministic connected target and
    # realistic sampled backend properties to Aer.
    from qiskit.providers.fake_provider import GenericBackendV2

    return GenericBackendV2(
        num_qubits=int(num_qubits),
        coupling_map=_linear_chain_coupling(int(num_qubits)),
        seed=int(seed),
        noise_info=True,
    )


@dataclass
class SelectedDevice:
    id: str
    source: str
    qubits: int
    coupling_map: list[list[int]]
    calibration_timestamp: str
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class LocalRuntime:
    """Qiskit Aer runner used only when Haiqu credentials are unavailable."""

    def __init__(
        self,
        device_id: str,
        seed: int,
        optimization_level: int = 2,
        qubit_count: int = 4,
    ):
        self.device_id = device_id
        self.seed = int(seed)
        self.optimization_level = int(optimization_level)
        self.qubit_count = int(qubit_count)
        self.fake_backend = None
        self.noise_model = None
        if device_id == "aer_simulator":
            coupling = _linear_chain_coupling(self.qubit_count)
            self.device = SelectedDevice(
                id=device_id,
                source="local_aer",
                qubits=self.qubit_count,
                coupling_map=coupling,
                calibration_timestamp=datetime.now(timezone.utc).isoformat(),
                raw_metadata={"simulator": True, "noise": "none"},
            )
            self.simulator = AerSimulator(seed_simulator=self.seed)
        else:
            self.fake_backend = _fake_backend(
                device_id, self.seed, num_qubits=self.qubit_count
            )
            self.noise_model = NoiseModel.from_backend(self.fake_backend)
            coupling = [
                [int(a), int(b)] for a, b in self.fake_backend.coupling_map.get_edges()
            ]
            timestamp = getattr(self.fake_backend, "dtm", None)
            self.device = SelectedDevice(
                id=device_id,
                source="local_qiskit_fake_backend",
                qubits=int(self.fake_backend.num_qubits),
                coupling_map=coupling,
                calibration_timestamp=(
                    timestamp.isoformat()
                    if hasattr(timestamp, "isoformat")
                    else datetime.now(timezone.utc).isoformat()
                ),
                raw_metadata={
                    "backend_class": type(self.fake_backend).__name__,
                    "simulator": True,
                    "noise_model_source": "qiskit_aer.NoiseModel.from_backend",
                },
            )
            self.simulator = AerSimulator(
                noise_model=self.noise_model,
                seed_simulator=self.seed,
            )

    def subgraph(self) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        return choose_connected_subgraph(
            self.device.coupling_map, size=self.qubit_count
        )

    def _compact_coupling(
        self, physical_qubits: Sequence[int], tree_edges: Sequence[Sequence[int]]
    ) -> CouplingMap:
        mapping = {int(qubit): index for index, qubit in enumerate(physical_qubits)}
        directed = []
        for left, right in tree_edges:
            a, b = mapping[int(left)], mapping[int(right)]
            directed.extend([(a, b), (b, a)])
        return CouplingMap(directed)

    def transpile(
        self,
        circuits: Sequence[QuantumCircuit],
        *,
        physical_qubits: Sequence[int],
        tree_edges: Sequence[Sequence[int]],
    ) -> list[QuantumCircuit]:
        if self.device_id == "aer_simulator":
            return list(
                transpile(
                    circuits,
                    backend=self.simulator,
                    optimization_level=self.optimization_level,
                    seed_transpiler=self.seed,
                )
            )
        compact = self._compact_coupling(physical_qubits, tree_edges)
        return list(
            transpile(
                circuits,
                basis_gates=self.noise_model.basis_gates,
                coupling_map=compact,
                optimization_level=self.optimization_level,
                initial_layout=list(range(len(physical_qubits))),
                seed_transpiler=self.seed,
            )
        )

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        *,
        shots: int,
        seed: int | None = None,
    ) -> list[dict[str, int]]:
        execution_seed = self.seed if seed is None else int(seed)
        result = self.simulator.run(
            list(circuits),
            shots=int(shots),
            seed_simulator=execution_seed,
        ).result()
        counts = result.get_counts()
        return (
            [dict(counts)] if isinstance(counts, Mapping) else [dict(x) for x in counts]
        )

    def run_ideal(
        self,
        circuits: Sequence[QuantumCircuit],
        *,
        shots: int,
    ) -> list[dict[str, int]]:
        result = (
            AerSimulator(seed_simulator=self.seed)
            .run(
                list(circuits),
                shots=int(shots),
                seed_simulator=self.seed,
            )
            .result()
        )
        counts = result.get_counts()
        return (
            [dict(counts)] if isinstance(counts, Mapping) else [dict(x) for x in counts]
        )


class HaiquSession:
    """Thin adapter over the inspected haiqu-sdk 1.3.1 public API."""

    def __init__(self, config: ExecutionConfig, sdk: Any | None = None):
        self.config = config
        self.sdk = sdk
        self.cloud_enabled = False
        self.cloud_unavailable_reason: str | None = None
        self.experiments: dict[str, dict[str, Any]] = {}
        self.device_model = None
        self.selected_device: SelectedDevice | None = None
        self.local_runtime: LocalRuntime | None = None
        try:
            self.sdk_version = metadata.version("haiqu-sdk")
        except metadata.PackageNotFoundError:
            self.sdk_version = "not-installed"

    def authenticate(self) -> bool:
        self.config.validate()
        if not self.config.haiqu_api_key:
            self.cloud_unavailable_reason = "HAIQU_API_KEY is not set"
            if not self.config.allow_local_fallback:
                raise RuntimeError(self.cloud_unavailable_reason)
            return False
        if self.sdk is None:
            try:
                from haiqu.sdk import haiqu
            except ImportError as exc:
                self.cloud_unavailable_reason = (
                    "HAIQU_API_KEY is set but haiqu-sdk is not installed"
                )
                if self.config.require_cloud or not self.config.allow_local_fallback:
                    raise RuntimeError(self.cloud_unavailable_reason) from exc
                return False
            self.sdk = haiqu
        try:
            status = self.sdk.login(api_access_key=self.config.haiqu_api_key)
        except Exception as exc:
            self.cloud_unavailable_reason = (
                f"Haiqu login failed: {type(exc).__name__}: {exc}"
            )
            if self.config.require_cloud or not self.config.allow_local_fallback:
                raise RuntimeError(self.cloud_unavailable_reason) from exc
            return False
        if not str(status).startswith("Success:"):
            self.cloud_unavailable_reason = f"Haiqu login failed: {status}"
            if self.config.require_cloud or not self.config.allow_local_fallback:
                raise RuntimeError(self.cloud_unavailable_reason)
            return False
        self.cloud_enabled = True
        self.activate_experiment("root")
        return True

    def activate_experiment(self, group: str) -> dict[str, Any] | None:
        if not self.cloud_enabled:
            return None
        name = EXPERIMENT_GROUPS[group]
        status = self.sdk.init(name)
        match = re.search(r"https://\S+", str(status))
        model = getattr(self.sdk, "_experiment", None)
        record = {
            "name": name,
            "id": getattr(model, "id", None),
            "dashboard_url": match.group(0) if match else None,
            "status": str(status),
        }
        self.experiments[group] = record
        return record

    @staticmethod
    def _as_device_list(value: Any) -> list:
        if value is None:
            return []
        if isinstance(value, pd.DataFrame):
            return value.to_dict("records")
        return list(value)

    def select_device(self, requested: str) -> SelectedDevice:
        if self.cloud_enabled:
            devices = self._as_device_list(
                self.sdk.list_devices(widget=False, pandas=False)
            )
            simulators = self._as_device_list(
                self.sdk.list_simulators(widget=False, pandas=False)
            )
            available = {
                str(
                    getattr(
                        item, "id", item.get("id") if isinstance(item, dict) else ""
                    )
                ): item
                for item in devices + simulators
            }
            fallback_order = (requested, "fake_fez", "aer_simulator")
            selected_id = next((x for x in fallback_order if x in available), None)
            if selected_id is None:
                raise RuntimeError("No requested Haiqu execution target is available")
            self.device_model = self.sdk.get_device(selected_id)
            payload = (
                self.device_model.model_dump(mode="json")
                if hasattr(self.device_model, "model_dump")
                else dict(self.device_model)
            )
            self.selected_device = SelectedDevice(
                id=selected_id,
                source="haiqu_cloud",
                qubits=int(payload["qubits"]),
                coupling_map=[
                    list(map(int, edge)) for edge in payload.get("coupling_map") or []
                ],
                calibration_timestamp=str(
                    payload.get("last_updated")
                    or datetime.now(timezone.utc).isoformat()
                ),
                raw_metadata=payload,
            )
            return self.selected_device

        try:
            self.local_runtime = LocalRuntime(
                requested,
                self.config.seed,
                self.config.optimization_level,
                self.config.demo_qubit_count,
            )
        except ValueError:
            self.local_runtime = LocalRuntime(
                "fake_fez",
                self.config.seed,
                self.config.optimization_level,
                self.config.demo_qubit_count,
            )
        self.selected_device = self.local_runtime.device
        return self.selected_device

    def subgraph(self) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
        if self.selected_device is None:
            raise RuntimeError("select_device must be called first")
        return choose_connected_subgraph(
            self.selected_device.coupling_map, size=self.config.demo_qubit_count
        )

    def log_object(self, obj: Any, *, name: str, group: str) -> Any | None:
        if not self.cloud_enabled:
            return None
        self.activate_experiment(group)
        return self.sdk.log(obj, name=name)

    def log_circuits(
        self, circuits: Sequence[QuantumCircuit], *, group: str = "diagnostics"
    ) -> list[Any]:
        if not self.cloud_enabled:
            return []
        self.activate_experiment(group)
        return [self.sdk.log(circuit, name=circuit.name) for circuit in circuits]

    def transpile_cloud(
        self,
        logged_circuits: Sequence[Any],
        *,
        group: str = "diagnostics",
    ) -> tuple[list[Any], pd.DataFrame]:
        if not self.cloud_enabled or self.device_model is None:
            raise RuntimeError("Haiqu cloud/device is not initialized")
        self.activate_experiment(group)
        result = self.sdk.transpile(
            list(logged_circuits),
            self.device_model,
            seed_transpiler=[self.config.seed, self.config.seed + 1],
            optimization_level=self.config.optimization_level,
        )
        models = result if isinstance(result, list) else [result]
        rows = []
        for model in models:
            model.wait_for_analytics(widget=False)
            analytics = model.analytics.model_dump() if model.analytics else {}
            rows.append(
                {
                    "circuit_name": model.name,
                    "stage": "transpiled",
                    "device_id": self.selected_device.id,
                    "depth": analytics.get("depth"),
                    "two_qubit_depth": analytics.get("depth_2q"),
                    "two_qubit_gate_count": analytics.get("gates_2q"),
                    "active_qubits": analytics.get("num_qubits_active"),
                    "mapped_physical_qubits": None,
                    "estimated_fidelity": (model.metrics or {}).get(
                        "estimated_fidelity"
                    ),
                    "estimated_survival_rate": (model.metrics or {}).get(
                        "estimated_survival_rate"
                    ),
                    "estimated_cost_usd": None,
                    "analytics_source": "haiqu_cloud",
                    "haiqu_circuit_id": model.id,
                }
            )
        return models, pd.DataFrame(rows)

    def run_cloud(
        self,
        circuits: Sequence[Any],
        *,
        mode: MitigationMode,
        group: str,
        job_name: str,
    ) -> tuple[list[dict[str, float]], Any]:
        if not self.cloud_enabled or self.device_model is None:
            raise RuntimeError("Haiqu cloud/device is not initialized")
        self.activate_experiment(group)
        options = mode.run_options
        if self.selected_device.id.lower().startswith("ibm_"):
            options = {**options, **self.config.ibm_credentials}
        job = self.sdk.run(
            circuits=list(circuits),
            shots=self.config.shots,
            device=self.device_model,
            options=options,
            use_mitigation=mode.use_mitigation,
            job_name=job_name,
        )
        return job.result(), job

    def hybrid_program(self, mode: MitigationMode):
        """Build a validated SDK 1.3.1 flow; custom inference remains modular."""

        if self.selected_device is None:
            raise RuntimeError("select_device must be called first")
        from haiqu.sdk.hybrid import HybridProgram, layers

        program_layers: list[Any] = [
            layers.InputLayer(),
            layers.TranspilationLayer(
                optimization_level=self.config.optimization_level
            ),
        ]
        if mode is not MitigationMode.RAW:
            options = mode.error_mitigation_options
            program_layers.append(
                layers.DistributionMitigationLayer(
                    mitigation_enabled=True,
                    advanced_mitigation=options.get("advanced_mitigation", True),
                    readout_mitigation=options.get("readout_mitigation", True),
                    noise_tailoring=options.get("noise_tailoring", False),
                    dynamical_decoupling=options.get("dynamical_decoupling", True),
                )
            )
        program_layers.append(layers.DeviceLayer(device_id=self.selected_device.id))
        return HybridProgram(layers=program_layers)

    def local_transpile_and_run(
        self,
        circuits: Sequence[QuantumCircuit],
        *,
        physical_qubits: Sequence[int],
        tree_edges: Sequence[Sequence[int]],
        seed: int | None = None,
    ) -> tuple[list[QuantumCircuit], list[dict[str, int]], pd.DataFrame]:
        if self.local_runtime is None:
            raise RuntimeError("Local runtime is not initialized")
        transpiled_circuits = self.local_runtime.transpile(
            circuits,
            physical_qubits=physical_qubits,
            tree_edges=tree_edges,
        )
        counts = self.local_runtime.run(
            transpiled_circuits,
            shots=self.config.shots,
            seed=seed,
        )
        rows = [
            local_circuit_analytics(
                circuit,
                stage="transpiled",
                device_id=self.selected_device.id,
                physical_qubits=(original.metadata or {}).get("physical_qubits"),
            )
            for original, circuit in zip(circuits, transpiled_circuits)
        ]
        return transpiled_circuits, counts, pd.DataFrame(rows)

    def manifest(self) -> dict[str, Any]:
        return {
            "haiqu_sdk_version": self.sdk_version,
            "cloud_enabled": self.cloud_enabled,
            "cloud_unavailable_reason": self.cloud_unavailable_reason,
            "experiments": self.experiments,
                "device": (
                {
                    "id": self.selected_device.id,
                    "source": self.selected_device.source,
                    "qubits": self.selected_device.qubits,
                    "calibration_timestamp": self.selected_device.calibration_timestamp,
                    "metadata": self.selected_device.raw_metadata,
                }
                if self.selected_device
                else None
            ),
            "transpilation": {
                "optimization_level": self.config.optimization_level,
                "seed_transpiler": self.config.seed,
            },
            "api_features": {
                "login": True,
                "experiment_tracking": True,
                "circuit_logging": True,
                "cloud_transpilation": True,
                "run": True,
                "mitigation_flags": [
                    "dynamical_decoupling",
                    "readout_mitigation",
                    "noise_tailoring",
                    "advanced_mitigation",
                ],
                "vector_loading": True,
                "hybrid_flow": True,
                "custom_model_inference_layer": False,
            },
        }
