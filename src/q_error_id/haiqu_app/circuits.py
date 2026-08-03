"""Diagnostic and benchmark circuit construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from math import ceil, log2
from typing import Any

import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit

PROBE_STATES_1Q = ("X+", "X-", "Y+", "Y-", "Z+", "Z-")
PAULI_MEASUREMENTS_1Q = ("X", "Y", "Z")
PROBE_STATES_2Q = tuple(product(("Z+", "Z-", "X+", "Y+"), repeat=2))
PAULI_MEASUREMENTS_2Q = ("ZI", "IZ", "ZX", "ZZ")
REQUIRED_METADATA = {
    "gate_name",
    "physical_qubits",
    "probe_state",
    "measurement_basis",
    "feature_index",
    "calibration_round",
}


@dataclass(frozen=True)
class ProtocolSpec:
    """Minimal Agent 1 protocol view used by the circuit layer."""

    name: str
    n_qubits: int
    input_labels: tuple[tuple[str, ...], ...]
    observable_labels: tuple[str, ...]
    settings: tuple[tuple[int, int], ...]

    @property
    def feature_count(self) -> int:
        return len(self.settings)

    @property
    def feature_labels(self) -> tuple[str, ...]:
        return tuple(
            f"{','.join(self.input_labels[i])}->{self.observable_labels[j]}"
            for i, j in self.settings
        )


def one_qubit_protocol_spec() -> ProtocolSpec:
    """Load Agent 1's contract when present, otherwise use its fixed 18 settings."""

    try:
        from q_error_id.core.protocols import one_qubit_protocol

        protocol = one_qubit_protocol()
        return ProtocolSpec(
            name=protocol.name,
            n_qubits=protocol.n_qubits,
            input_labels=tuple(tuple(x) for x in protocol.input_labels),
            observable_labels=tuple(protocol.observable_labels),
            settings=tuple(tuple(x) for x in protocol.settings),
        )
    except ImportError:
        labels = tuple((label,) for label in PROBE_STATES_1Q)
        settings = tuple(product(range(6), range(3)))
        return ProtocolSpec(
            "one_qubit_six_state_pauli",
            1,
            labels,
            PAULI_MEASUREMENTS_1Q,
            settings,
        )


def two_qubit_protocol_spec() -> ProtocolSpec:
    """Load Agent 1's selected protocol or a compatible 64-feature bank."""

    try:
        from q_error_id.core.protocols import two_qubit_protocol

        protocol = two_qubit_protocol(
            gate_name="CX",
            basis=("ZI", "IZ", "ZX", "ZZ"),
            target_features=80,
        )
        return ProtocolSpec(
            name=protocol.name,
            n_qubits=protocol.n_qubits,
            input_labels=tuple(tuple(x) for x in protocol.input_labels),
            observable_labels=tuple(protocol.observable_labels),
            settings=tuple(tuple(x) for x in protocol.settings),
        )
    except ImportError:
        settings = tuple(
            product(
                range(len(PROBE_STATES_2Q)),
                range(len(PAULI_MEASUREMENTS_2Q)),
            )
        )
        return ProtocolSpec(
            "two_qubit_product_64_fallback",
            2,
            PROBE_STATES_2Q,
            PAULI_MEASUREMENTS_2Q,
            settings,
        )


def _prepare_state(circuit: QuantumCircuit, qubit: int, label: str) -> None:
    if label == "X+":
        circuit.h(qubit)
    elif label == "X-":
        circuit.x(qubit)
        circuit.h(qubit)
    elif label == "Y+":
        circuit.h(qubit)
        circuit.s(qubit)
    elif label == "Y-":
        circuit.h(qubit)
        circuit.sdg(qubit)
    elif label == "Z+":
        return
    elif label == "Z-":
        circuit.x(qubit)
    else:
        raise ValueError(f"Unsupported probe state: {label}")


def _apply_measurement_rotation(
    circuit: QuantumCircuit, qubit: int, pauli: str
) -> None:
    if pauli == "X":
        circuit.h(qubit)
    elif pauli == "Y":
        circuit.sdg(qubit)
        circuit.h(qubit)
    elif pauli in {"Z", "I"}:
        return
    else:
        raise ValueError(f"Unsupported measurement basis: {pauli}")


def _apply_one_qubit_gate(circuit: QuantumCircuit, qubit: int, gate_name: str) -> None:
    normalized = gate_name.lower()
    if normalized == "id":
        circuit.id(qubit)
    elif normalized == "x":
        circuit.x(qubit)
    elif normalized == "sx":
        circuit.sx(qubit)
    elif normalized == "h":
        circuit.h(qubit)
    else:
        raise ValueError(f"Unsupported diagnostic 1Q gate: {gate_name}")


def validate_diagnostic_metadata(circuit: QuantumCircuit) -> None:
    metadata = circuit.metadata or {}
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ValueError(f"Diagnostic circuit metadata is missing: {sorted(missing)}")
    if int(metadata["feature_index"]) < 0:
        raise ValueError("feature_index must be nonnegative")
    if not metadata["physical_qubits"]:
        raise ValueError("physical_qubits cannot be empty")


def build_one_qubit_diagnostics(
    physical_qubit: int,
    *,
    gate_name: str = "id",
    calibration_round: int = 0,
    width: int = 1,
    circuit_qubit: int = 0,
    protocol: ProtocolSpec | None = None,
) -> list[QuantumCircuit]:
    """Build the Agent 1 six-state Pauli protocol as measured Qiskit circuits."""

    protocol = protocol or one_qubit_protocol_spec()
    circuits: list[QuantumCircuit] = []
    for feature_index, (input_index, observable_index) in enumerate(protocol.settings):
        probe = protocol.input_labels[input_index][0]
        observable = protocol.observable_labels[observable_index]
        circuit = QuantumCircuit(width, width)
        _prepare_state(circuit, circuit_qubit, probe)
        circuit.barrier()
        _apply_one_qubit_gate(circuit, circuit_qubit, gate_name)
        circuit.barrier()
        _apply_measurement_rotation(circuit, circuit_qubit, observable)
        circuit.measure(range(width), range(width))
        circuit.name = f"diag_1q_q{physical_qubit}_f{feature_index}"
        circuit.metadata = {
            "protocol_name": protocol.name,
            "channel_key": f"q{physical_qubit}",
            "gate_name": gate_name,
            "physical_qubits": [int(physical_qubit)],
            "circuit_qubits": [int(circuit_qubit)],
            "measurement_clbits": [int(circuit_qubit)],
            "probe_state": probe,
            "measurement_basis": observable,
            "feature_index": feature_index,
            "feature_label": protocol.feature_labels[feature_index],
            "calibration_round": int(calibration_round),
        }
        validate_diagnostic_metadata(circuit)
        circuits.append(circuit)
    return circuits


def build_two_qubit_diagnostics(
    physical_qubits: tuple[int, int],
    *,
    gate_name: str = "cx",
    calibration_round: int = 0,
    width: int = 2,
    circuit_qubits: tuple[int, int] = (0, 1),
    protocol: ProtocolSpec | None = None,
) -> list[QuantumCircuit]:
    """Build the configured product-state/CX-like diagnostic bank."""

    protocol = protocol or two_qubit_protocol_spec()
    circuits: list[QuantumCircuit] = []
    for feature_index, (input_index, observable_index) in enumerate(protocol.settings):
        probe = protocol.input_labels[input_index]
        observable = protocol.observable_labels[observable_index]
        circuit = QuantumCircuit(width, width)
        for qubit, label in zip(circuit_qubits, probe):
            _prepare_state(circuit, qubit, label)
        circuit.barrier()
        if gate_name.lower() != "cx":
            raise ValueError("The integrated two-qubit protocol currently targets CX")
        circuit.cx(*circuit_qubits)
        circuit.barrier()
        for qubit, pauli in zip(circuit_qubits, observable):
            _apply_measurement_rotation(circuit, qubit, pauli)
        circuit.measure(range(width), range(width))
        edge = f"q{physical_qubits[0]}-q{physical_qubits[1]}"
        circuit.name = f"diag_2q_{edge}_f{feature_index}"
        circuit.metadata = {
            "protocol_name": protocol.name,
            "channel_key": edge,
            "gate_name": gate_name,
            "physical_qubits": [int(q) for q in physical_qubits],
            "circuit_qubits": [int(q) for q in circuit_qubits],
            "measurement_clbits": [int(q) for q in circuit_qubits],
            "probe_state": ",".join(probe),
            "measurement_basis": observable,
            "feature_index": feature_index,
            "feature_label": protocol.feature_labels[feature_index],
            "calibration_round": int(calibration_round),
        }
        validate_diagnostic_metadata(circuit)
        circuits.append(circuit)
    return circuits


def build_readout_calibration_circuits(
    physical_qubits: Sequence[int],
    edges: Sequence[Sequence[int]],
    *,
    width: int = 4,
    calibration_round: int = 0,
) -> list[QuantumCircuit]:
    """Build 2-state node and 4-state edge assignment calibrations.

    Every circuit measures the full compact register so the same transpilation
    layout and readout channel are exercised as in the diagnostic bank.
    """

    qubits = tuple(int(qubit) for qubit in physical_qubits)
    if len(qubits) != width or len(set(qubits)) != width:
        raise ValueError("physical_qubits must contain one entry per compact qubit")
    compact_index = {physical: logical for logical, physical in enumerate(qubits)}
    circuits: list[QuantumCircuit] = []

    def build(
        *,
        key: str,
        selected_physical: tuple[int, ...],
        prepared_state: str,
    ) -> QuantumCircuit:
        selected_compact = tuple(compact_index[q] for q in selected_physical)
        circuit = QuantumCircuit(width, width)
        for logical_index, circuit_qubit in enumerate(selected_compact):
            if prepared_state[-1 - logical_index] == "1":
                circuit.x(circuit_qubit)
        circuit.barrier()
        circuit.measure(range(width), range(width))
        compact_state = prepared_state.replace(" ", "")
        circuit.name = (
            f"readout_cal_{key.replace('-', '_')}_prep_{compact_state}"
        )
        circuit.metadata = {
            "calibration_type": "readout_assignment",
            "calibration_key": key,
            "physical_qubits": list(selected_physical),
            "circuit_qubits": list(selected_compact),
            "measurement_clbits": list(selected_compact),
            "prepared_state": compact_state,
            "bitstring_order": [
                format(index, f"0{len(selected_physical)}b")
                for index in range(2 ** len(selected_physical))
            ],
            "calibration_round": int(calibration_round),
        }
        return circuit

    for physical in qubits:
        for prepared in ("0", "1"):
            circuits.append(
                build(
                    key=f"q{physical}",
                    selected_physical=(physical,),
                    prepared_state=prepared,
                )
            )
    for raw_left, raw_right in edges:
        left, right = int(raw_left), int(raw_right)
        for prepared in ("00", "01", "10", "11"):
            circuits.append(
                build(
                    key=f"q{left}-q{right}",
                    selected_physical=(left, right),
                    prepared_state=prepared,
                )
            )
    return circuits


def diagnostic_table(circuits: Iterable[QuantumCircuit]) -> pd.DataFrame:
    rows = []
    for circuit in circuits:
        validate_diagnostic_metadata(circuit)
        metadata = circuit.metadata or {}
        rows.append(
            {
                "circuit_name": circuit.name,
                **{
                    key: metadata[key]
                    for key in (
                        "channel_key",
                        "gate_name",
                        "physical_qubits",
                        "probe_state",
                        "measurement_basis",
                        "feature_index",
                        "calibration_round",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def local_circuit_analytics(
    circuit: QuantumCircuit,
    *,
    stage: str,
    device_id: str,
    physical_qubits: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Collect the requested circuit metrics without inventing cost data."""

    active = set()
    two_qubit_count = 0
    for instruction in circuit.data:
        if instruction.operation.name in {"measure", "barrier"}:
            continue
        active.update(circuit.find_bit(q).index for q in instruction.qubits)
        if len(instruction.qubits) == 2:
            two_qubit_count += 1
    try:
        depth_2q = circuit.depth(filter_function=lambda item: len(item.qubits) == 2)
    except TypeError:
        depth_2q = two_qubit_count
    metadata = circuit.metadata or {}
    mapped = list(physical_qubits or metadata.get("physical_qubits") or sorted(active))
    return {
        "circuit_name": circuit.name,
        "stage": stage,
        "device_id": device_id,
        "depth": circuit.depth(),
        "two_qubit_depth": depth_2q,
        "two_qubit_gate_count": two_qubit_count,
        "active_qubits": len(active),
        "mapped_physical_qubits": mapped,
        "estimated_fidelity": None,
        "estimated_survival_rate": None,
        "estimated_cost_usd": None,
        "analytics_source": "local_qiskit",
    }


def build_benchmark_circuit(width: int = 4) -> QuantumCircuit:
    """Create a four-qubit GHZ benchmark with idle-sensitive motifs."""

    if width != 4:
        raise ValueError("The benchmark is defined for four qubits")
    circuit = QuantumCircuit(width, width, name="q_error_id_4q_benchmark")
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.delay(160, 2, unit="dt")
    circuit.delay(160, 3, unit="dt")
    circuit.cx(1, 2)
    circuit.delay(160, 0, unit="dt")
    circuit.cx(2, 3)
    circuit.barrier()
    circuit.measure(range(width), range(width))
    circuit.metadata = {
        "gate_name": "four_qubit_ghz",
        "physical_qubits": list(range(width)),
        "probe_state": "0000",
        "measurement_basis": "ZZZZ",
        "feature_index": 0,
        "calibration_round": 0,
        "known_success_states": ["0000", "1111"],
    }
    return circuit


def build_angle_encoded_inference_circuit(
    features: Sequence[float],
    *,
    max_qubits: int = 8,
) -> QuantumCircuit:
    """Build the non-opaque fallback inference encoding used without an amplitude QNN."""

    values = [float(value) for value in features]
    num_qubits = min(max_qubits, max(1, len(values)))
    circuit = QuantumCircuit(num_qubits, name="angle_encoded_inference")
    for index, value in enumerate(values):
        qubit = index % num_qubits
        circuit.ry(value, qubit)
        if index >= num_qubits:
            circuit.rz(0.5 * value, qubit)
    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)
    circuit.metadata = {
        "encoding": "angle",
        "feature_count": len(values),
        "head": "fixed_demo_entangling_head",
    }
    return circuit


def build_state_preparation_circuit(features: Sequence[float]) -> QuantumCircuit:
    """Build ordinary Qiskit amplitude preparation for Haiqu comparison."""

    from qiskit.circuit.library import StatePreparation

    values = np.asarray(features, dtype=complex).reshape(-1)
    num_qubits = max(1, ceil(log2(max(values.size, 1))))
    padded = np.zeros(2**num_qubits, dtype=complex)
    padded[: values.size] = values
    norm = np.linalg.norm(padded)
    if norm == 0.0:
        padded[0] = 1.0
    else:
        padded /= norm
    circuit = QuantumCircuit(num_qubits, name="qiskit_state_preparation")
    circuit.append(StatePreparation(padded), range(num_qubits))
    circuit.metadata = {
        "encoding": "qiskit_state_preparation",
        "feature_count": values.size,
    }
    return circuit


def _append_edge_correction(
    circuit: QuantumCircuit,
    qubits: tuple[int, int],
    label: str,
    alpha: float,
) -> None:
    """Append the first-order inverse of exp(-i alpha P / 2)."""

    q0, q1 = qubits
    theta = -float(alpha)
    if label == "ZI":
        circuit.rz(theta, q0)
    elif label == "IZ":
        circuit.rz(theta, q1)
    elif label == "ZZ":
        circuit.rzz(theta, q0, q1)
    elif label == "ZX":
        circuit.h(q1)
        circuit.rzz(theta, q0, q1)
        circuit.h(q1)


def apply_coherent_correction(
    benchmark: QuantumCircuit,
    single_qubit_channels: dict[int, dict],
    two_qubit_channels: dict[str, dict],
) -> QuantumCircuit:
    """Return a prototype circuit with learned coherent inverse rotations.

    The added rotations represent ideal calibration/frame updates in this
    hackathon prototype. Their native-device error must be included before any
    deployment claim.
    """

    corrected = benchmark.remove_final_measurements(inplace=False)
    if corrected.num_clbits < corrected.num_qubits:
        corrected.add_register(ClassicalRegister(corrected.num_qubits, "c_corr"))
    for qubit, channel in sorted(single_qubit_channels.items()):
        alpha = channel.get("alpha", {})
        corrected.rx(-float(alpha.get("X", 0.0)), qubit)
        corrected.ry(-float(alpha.get("Y", 0.0)), qubit)
        corrected.rz(-float(alpha.get("Z", 0.0)), qubit)
    for edge, channel in sorted(two_qubit_channels.items()):
        physical = channel.get("physical_qubits")
        if not physical:
            physical = tuple(int(x[1:]) for x in edge.split("-"))
        for label, alpha in channel.get("alpha", {}).items():
            _append_edge_correction(
                corrected,
                tuple(int(x) for x in physical),
                label,
                float(alpha),
            )
    corrected.measure(range(corrected.num_qubits), range(corrected.num_qubits))
    corrected.name = f"{benchmark.name}_corrected"
    corrected.metadata = dict(benchmark.metadata or {})
    corrected.metadata["coherent_correction"] = True
    corrected.metadata["correction_semantics"] = "ideal_calibration_update_prototype"
    return corrected
