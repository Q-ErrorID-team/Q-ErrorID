"""A realistic, training-independent demo: QROM 'phonebook' lookup correction.

This mirrors the 3-index-qubit / 2-data-qubit telephone-book example from the
QML2026 day1 "Data Loading" notebook. It is deliberately kept separate from
the Grover training/validation loop: it consumes already-reconstructed local
channel estimates (from any prior diagnostic run) and demonstrates Raw /
Readout-only / Learned-correction on a single, deterministic QROM lookup.

Unlike the Grover benchmark, a single lookup circuit has exactly one intended
answer -- there is no set of alternative "targets" to build an invertible
response matrix from actual runs. We recover that structure the same way the
Grover benchmark does: by *forward-simulating*, under the reconstructed noise
model, what the same physical gate sequence would produce for each of the
four hypothetical 2-bit values the QROM write step could have encoded. Only
one of those four is ever physically executed (the true phonebook entry);
the other three exist purely in simulation, exactly as Grover forward-
simulates all four candidate targets while only one is measured per run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Kraus, Operator
from qiskit.transpiler import CouplingMap

from q_error_id.core import (
    build_channel,
    channel_to_kraus,
    one_qubit_parameters,
    two_qubit_parameters,
)

from .algorithm_benchmark import project_probability_simplex
from .readout import marginalize_distribution

# The 8-entry call history from the day1 Data Loading demo notebook
# (3 index qubits address 8 entries; each entry is a 2-bit data value).
PHONEBOOK: tuple[str, ...] = ("10", "01", "01", "00", "11", "01", "11", "10")
DATA_OUTCOMES: tuple[str, ...] = ("00", "01", "10", "11")
ONE_Q_LABELS = ("X", "Y", "Z")
TWO_Q_LABELS = ("ZI", "IZ", "ZX", "ZZ")


def _channel_values(channel: Mapping[str, Any], field: str, labels: Sequence[str]) -> np.ndarray:
    values = channel.get(field, {})
    return np.asarray([float(values.get(label, 0.0)) for label in labels], dtype=float)


def _single_kappa(channel: Mapping[str, Any]) -> float:
    values = channel.get("kappa_down", {})
    if isinstance(values, Mapping):
        return float(max((float(value) for value in values.values()), default=0.0))
    array = np.asarray(values, dtype=float).reshape(-1)
    return float(array[0]) if array.size else 0.0


def build_phonebook_lookup_circuit(
    index_bits: str,
    data_bits: str,
    *,
    index_qubits: tuple[int, int, int],
    data_qubits: tuple[int, int],
    width: int,
    measure: bool = True,
) -> QuantumCircuit:
    """Deterministically prepare one index and write one hypothetical value.

    With the index register in a computational-basis state (not a Walsh-
    Hadamard superposition), only the single write-block matching that index
    can fire, so this reproduces one QROM lookup with a single deterministic
    ideal answer -- a small (<=8 logical gate), realistic, non-Grover circuit.
    """

    if len(index_bits) != len(index_qubits) or len(data_bits) != len(data_qubits):
        raise ValueError("index_bits/data_bits must match the qubit tuple lengths")

    circuit = QuantumCircuit(width, width if measure else 0)
    for qubit, bit in zip(index_qubits, index_bits):
        if bit == "1":
            circuit.x(qubit)
    circuit.barrier()
    for qubit, bit in zip(index_qubits, index_bits):
        if bit == "0":
            circuit.x(qubit)
    for data_qubit, bit in zip(data_qubits, data_bits):
        if bit == "1":
            circuit.mcx(list(index_qubits), data_qubit)
    for qubit, bit in zip(index_qubits, index_bits):
        if bit == "0":
            circuit.x(qubit)
    circuit.barrier()
    if measure:
        circuit.measure(range(width), range(width))
    circuit.name = f"phonebook_lookup_idx{index_bits}_write{data_bits}"
    circuit.metadata = {
        "algorithm": "phonebook_qrom_lookup",
        "index_bits": index_bits,
        "written_data_bits": data_bits,
        "index_qubits": list(index_qubits),
        "data_qubits": list(data_qubits),
        "physical_qubits": list(index_qubits) + list(data_qubits),
        "measurement_clbits": list(range(width)) if measure else [],
    }
    return circuit


def build_phonebook_superposition_circuit(
    *,
    index_qubits: tuple[int, int, int],
    data_qubits: tuple[int, int],
    width: int,
    measure: bool = True,
) -> QuantumCircuit:
    """Encode all 8 phonebook entries in superposition (the original notebook circuit).

    A Walsh-Hadamard transform puts the index register into an equal
    superposition of all 8 indices; the same X-select / mcx-write / X-unselect
    block used by :func:`build_phonebook_lookup_circuit` is then chained once
    per entry. Because only the branch matching each basis index has its
    controls all set, every branch independently and correctly writes its own
    data value, entangling index and data into
    ``(1/sqrt(8)) * sum_i |index_i> |data_i>``.
    """

    circuit = QuantumCircuit(width, width if measure else 0)
    for qubit in index_qubits:
        circuit.h(qubit)
    circuit.barrier()
    for index_value, data_bits in enumerate(PHONEBOOK):
        index_bits = format(index_value, f"0{len(index_qubits)}b")
        for qubit, bit in zip(index_qubits, index_bits):
            if bit == "0":
                circuit.x(qubit)
        for data_qubit, bit in zip(data_qubits, data_bits):
            if bit == "1":
                circuit.mcx(list(index_qubits), data_qubit)
        for qubit, bit in zip(index_qubits, index_bits):
            if bit == "0":
                circuit.x(qubit)
        circuit.barrier()
    if measure:
        circuit.measure(range(width), range(width))
    circuit.name = "phonebook_superposition_qrom"
    circuit.metadata = {
        "algorithm": "phonebook_qrom_superposition",
        "index_qubits": list(index_qubits),
        "data_qubits": list(data_qubits),
        "physical_qubits": list(index_qubits) + list(data_qubits),
        "measurement_clbits": list(range(width)) if measure else [],
        "phonebook": list(PHONEBOOK),
    }
    return circuit


def conditional_data_distributions(
    counts: Mapping[str, int | float],
    *,
    index_qubits: tuple[int, int, int],
    data_qubits: tuple[int, int],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Split a joint measurement into per-index conditional data distributions.

    Returns ``(conditional, index_marginal)`` where ``conditional[index_bits]``
    is the normalized data-outcome distribution measured whenever that index
    was observed, and ``index_marginal[index_bits]`` is the fraction of all
    shots that landed on that index (used to recombine corrected conditionals
    into a joint distribution).
    """

    index_width = len(index_qubits)
    grouped: dict[str, dict[str, float]] = {}
    index_totals: dict[str, float] = {}
    grand_total = 0.0
    for bitstring, weight in counts.items():
        clean = str(bitstring).replace(" ", "")[::-1]  # clean[i] == qubit i's bit
        weight = float(weight)
        grand_total += weight
        index_bits = "".join(clean[q] for q in index_qubits)
        data_bits = "".join(clean[q] for q in data_qubits)
        bucket = grouped.setdefault(index_bits, {outcome: 0.0 for outcome in DATA_OUTCOMES})
        bucket[data_bits] = bucket.get(data_bits, 0.0) + weight
        index_totals[index_bits] = index_totals.get(index_bits, 0.0) + weight

    conditional: dict[str, dict[str, float]] = {}
    for index_bits in (format(i, f"0{index_width}b") for i in range(2**index_width)):
        bucket = grouped.get(index_bits, {outcome: 0.0 for outcome in DATA_OUTCOMES})
        total = index_totals.get(index_bits, 0.0)
        conditional[index_bits] = (
            {outcome: value / total for outcome, value in bucket.items()}
            if total > 0
            else {outcome: 0.0 for outcome in DATA_OUTCOMES}
        )
    index_marginal = {
        index_bits: index_totals.get(index_bits, 0.0) / grand_total if grand_total else 0.0
        for index_bits in conditional
    }
    return conditional, index_marginal


def joint_distribution_from_conditionals(
    conditional: Mapping[str, Mapping[str, float]],
    index_marginal: Mapping[str, float],
) -> dict[str, float]:
    """Recombine per-index conditionals into one joint (index, data) distribution."""

    joint: dict[str, float] = {}
    for index_bits, weight in index_marginal.items():
        for data_bits, probability in conditional.get(index_bits, {}).items():
            joint[f"{index_bits}{data_bits}"] = weight * probability
    return joint


def ideal_joint_distribution() -> dict[str, float]:
    """The exact target distribution: each (index, data) pair at probability 1/8."""

    index_width = 3
    return {
        f"{format(i, f'0{index_width}b')}{data_bits}": 1.0 / len(PHONEBOOK)
        for i, data_bits in enumerate(PHONEBOOK)
    }


def _local_channel_kraus(
    channel: Mapping[str, Any] | None,
    physical_qubit: int,
) -> Kraus | None:
    if channel is None:
        return None
    params = one_qubit_parameters(
        alpha=_channel_values(channel, "alpha", ONE_Q_LABELS),
        gamma=_channel_values(channel, "gamma", ONE_Q_LABELS),
        kappa_down=_single_kappa(channel),
        qubit=physical_qubit,
    )
    return Kraus(list(channel_to_kraus(build_channel(params))))


def _edge_channel_kraus(
    channel: Mapping[str, Any] | None,
    edge: tuple[int, int],
) -> Kraus | None:
    if channel is None:
        return None
    params = two_qubit_parameters(
        gate_name="CX",
        alpha=_channel_values(channel, "alpha", TWO_Q_LABELS),
        gamma=_channel_values(channel, "gamma", TWO_Q_LABELS),
        kappa_down=np.zeros(2),
        qubits=edge,
        basis=TWO_Q_LABELS,
    )
    return Kraus(list(channel_to_kraus(build_channel(params))))


def simulate_lookup_with_generator(
    index_bits: str,
    data_bits: str,
    *,
    index_qubits: tuple[int, int, int],
    data_qubits: tuple[int, int],
    width: int,
    single_qubit_channels: Mapping[int, Mapping[str, Any]],
    two_qubit_channels: Mapping[tuple[int, int], Mapping[str, Any]],
    coupling_map: CouplingMap,
) -> dict[str, float]:
    """Forward-simulate one hypothetical QROM write under the learned noise model."""

    ideal = build_phonebook_lookup_circuit(
        index_bits,
        data_bits,
        index_qubits=index_qubits,
        data_qubits=data_qubits,
        width=width,
        measure=False,
    )
    transpiled = transpile(
        ideal,
        basis_gates=["x", "cx", "rz", "sx", "id"],
        coupling_map=coupling_map,
        optimization_level=1,
        initial_layout=list(range(width)),
        seed_transpiler=0,
    )

    state = DensityMatrix.from_label("0" * width)
    kraus_cache: dict[Any, Kraus | None] = {}
    for instruction in transpiled.data:
        operation = instruction.operation
        if operation.name == "barrier":
            continue
        qubit_indices = [transpiled.find_bit(qubit).index for qubit in instruction.qubits]
        state = state.evolve(Operator(operation), qargs=qubit_indices)
        if operation.name == "id":
            continue
        if len(qubit_indices) == 1:
            physical = qubit_indices[0]
            key = ("1q", physical)
            if key not in kraus_cache:
                kraus_cache[key] = _local_channel_kraus(
                    single_qubit_channels.get(physical), physical
                )
            kraus = kraus_cache[key]
            if kraus is not None:
                state = state.evolve(kraus, qargs=[physical])
        elif len(qubit_indices) == 2 and operation.name == "cx":
            edge = (qubit_indices[0], qubit_indices[1])
            key = ("2q", edge)
            if key not in kraus_cache:
                channel = two_qubit_channels.get(edge) or two_qubit_channels.get(
                    (edge[1], edge[0])
                )
                kraus_cache[key] = _edge_channel_kraus(channel, edge)
            kraus = kraus_cache[key]
            if kraus is not None:
                state = state.evolve(kraus, qargs=list(edge))

    # Routing may insert SWAPs, so the wire that started as a given virtual
    # qubit is not necessarily the wire holding its state at the end of the
    # circuit. `final_index_layout()[virtual] -> final wire` corrects for it.
    final_layout = transpiled.layout.final_index_layout()
    final_data_qubits = tuple(final_layout[q] for q in data_qubits)

    # marginalize_distribution((c0, c1)) returns "c1 c0"; reverse the tuple so
    # the returned string reads as data_qubits[0]'s bit followed by
    # data_qubits[1]'s bit, matching build_phonebook_lookup_circuit's
    # data_bits convention (and therefore the identity response at zero noise).
    probabilities = state.probabilities_dict()
    return marginalize_distribution(probabilities, tuple(reversed(final_data_qubits)))


@dataclass(frozen=True)
class PhonebookResponseModel:
    """Regularized response inverse for one QROM lookup instance.

    Built purely from the forward model (the four hypothetical write values),
    never from measured lookup counts -- the same no-fitting-on-the-test-data
    guarantee used by the Grover response model.
    """

    index_bits: str
    response_matrix: np.ndarray
    inverse_matrix: np.ndarray
    condition_number: float

    @classmethod
    def from_channels(
        cls,
        index_bits: str,
        *,
        index_qubits: tuple[int, int, int],
        data_qubits: tuple[int, int],
        width: int,
        single_qubit_channels: Mapping[int, Mapping[str, Any]],
        two_qubit_channels: Mapping[tuple[int, int], Mapping[str, Any]],
        coupling_map: CouplingMap,
        regularization: float = 0.03,
    ) -> PhonebookResponseModel:
        columns = []
        for data_bits in DATA_OUTCOMES:
            predicted = simulate_lookup_with_generator(
                index_bits,
                data_bits,
                index_qubits=index_qubits,
                data_qubits=data_qubits,
                width=width,
                single_qubit_channels=single_qubit_channels,
                two_qubit_channels=two_qubit_channels,
                coupling_map=coupling_map,
            )
            columns.append([predicted[outcome] for outcome in DATA_OUTCOMES])
        response = np.array(columns).T
        u_matrix, singular_values, vh_matrix = np.linalg.svd(response, full_matrices=False)
        factors = singular_values / (np.square(singular_values) + float(regularization))
        inverse = (vh_matrix.T * factors) @ u_matrix.T
        return cls(
            index_bits=index_bits,
            response_matrix=response,
            inverse_matrix=inverse,
            condition_number=float(np.linalg.cond(response)),
        )

    def correct(self, distribution: Mapping[str, float]) -> dict[str, float]:
        corrected, _audit = self.correct_with_audit(distribution)
        return corrected

    def correct_with_audit(
        self, distribution: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Apply the learned non-CPTP inverse and report projection diagnostics.

        Mirrors :meth:`GeneratorResponseModel.correct_with_audit` in
        ``algorithm_benchmark.py`` so the same validation-gate criteria
        (mean paired improvement, bounded negativity, bounded simplex
        projection cost) apply here too.
        """

        vector = np.asarray([distribution.get(o, 0.0) for o in DATA_OUTCOMES], dtype=float)
        total = float(vector.sum())
        if total > 0:
            vector = vector / total
        quasi = self.inverse_matrix @ vector
        corrected = project_probability_simplex(quasi)
        output = {outcome: float(corrected[i]) for i, outcome in enumerate(DATA_OUTCOMES)}
        audit = {
            "inverse_raw_normalization": float(quasi.sum()),
            "inverse_raw_negativity": float(np.abs(quasi[quasi < 0.0]).sum()),
            "simplex_projection_l1": float(np.linalg.norm(corrected - quasi, ord=1)),
        }
        return output, audit
