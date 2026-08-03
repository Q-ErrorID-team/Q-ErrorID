"""Choi-reference and reduced prepare-and-measure diagnostic protocols."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from .channels import QuantumChannel
from .pauli import I, pauli_word, product_state, state_from_label
from .representations import channel_to_choi


@dataclass(frozen=True)
class ReadoutConfusion:
    """Independent binary readout confusion for each measured qubit.

    ``p_plus_to_minus`` is the probability to report ``-1`` for a true
    ``+1`` outcome, and ``p_minus_to_plus`` is the reverse probability.
    Scalars apply to every qubit; vectors allow qubit-specific confusion.
    """

    p_plus_to_minus: float | Sequence[float] = 0.0
    p_minus_to_plus: float | Sequence[float] = 0.0

    def coefficients(self, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
        """Return affine coefficients ``a + b*s`` for a true eigenvalue ``s``."""

        plus_to_minus = np.broadcast_to(
            np.asarray(self.p_plus_to_minus, dtype=float), (n_qubits,)
        ).copy()
        minus_to_plus = np.broadcast_to(
            np.asarray(self.p_minus_to_plus, dtype=float), (n_qubits,)
        ).copy()
        if (
            np.any(plus_to_minus < 0.0)
            or np.any(minus_to_plus < 0.0)
            or np.any(plus_to_minus > 1.0)
            or np.any(minus_to_plus > 1.0)
        ):
            raise ValueError("Readout-confusion probabilities must lie in [0, 1]")
        offset = minus_to_plus - plus_to_minus
        scale = 1.0 - plus_to_minus - minus_to_plus
        return offset, scale

    def to_dict(self) -> dict[str, list[float] | float]:
        """Return JSON-compatible configuration values."""

        def serializable(value: float | Sequence[float]) -> list[float] | float:
            array = np.asarray(value, dtype=float)
            return float(array) if array.ndim == 0 else array.tolist()

        return {
            "p_plus_to_minus": serializable(self.p_plus_to_minus),
            "p_minus_to_plus": serializable(self.p_minus_to_plus),
        }


@dataclass(frozen=True)
class ChoiStateProtocol:
    """Exact system-ancilla reference protocol based on a normalized Choi state."""

    name: str = "choi_state_reference"


@dataclass(frozen=True)
class PrepareMeasureProtocol:
    """A bank of input density matrices and Pauli measurements."""

    name: str
    n_qubits: int
    input_labels: tuple[tuple[str, ...], ...]
    input_states: tuple[np.ndarray, ...]
    observable_labels: tuple[str, ...]
    observables: tuple[np.ndarray, ...]
    settings: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.input_labels) != len(self.input_states):
            raise ValueError("Input labels and states must have equal lengths")
        if len(self.observable_labels) != len(self.observables):
            raise ValueError("Observable labels and matrices must have equal lengths")
        for input_index, observable_index in self.settings:
            if not 0 <= input_index < len(self.input_states):
                raise ValueError("A setting refers to an unknown input")
            if not 0 <= observable_index < len(self.observables):
                raise ValueError("A setting refers to an unknown observable")

    @property
    def feature_count(self) -> int:
        return len(self.settings)

    @property
    def feature_labels(self) -> tuple[str, ...]:
        """Return stable ``input->observable`` labels."""

        return tuple(
            f"{','.join(self.input_labels[input_index])}->{self.observable_labels[observable_index]}"
            for input_index, observable_index in self.settings
        )

    def subset(
        self, indices: Sequence[int], name: str | None = None
    ) -> PrepareMeasureProtocol:
        """Return a protocol containing selected feature settings."""

        chosen = tuple(self.settings[int(index)] for index in indices)
        return PrepareMeasureProtocol(
            name=name or f"{self.name}_subset",
            n_qubits=self.n_qubits,
            input_labels=self.input_labels,
            input_states=self.input_states,
            observable_labels=self.observable_labels,
            observables=self.observables,
            settings=chosen,
        )


def one_qubit_protocol() -> PrepareMeasureProtocol:
    """Return the fixed six-state, three-observable, 18-feature protocol."""

    labels = (("X+",), ("X-",), ("Y+",), ("Y-",), ("Z+",), ("Z-",))
    states = tuple(state_from_label(label[0]) for label in labels)
    observable_labels = ("X", "Y", "Z")
    observables = tuple(pauli_word(label) for label in observable_labels)
    settings = tuple(product(range(len(states)), range(len(observables))))
    return PrepareMeasureProtocol(
        name="one_qubit_six_state_pauli",
        n_qubits=1,
        input_labels=labels,
        input_states=states,
        observable_labels=observable_labels,
        observables=observables,
        settings=settings,
    )


def two_qubit_candidate_protocol(
    local_input_labels: tuple[str, ...] = ("Z+", "Z-", "X+", "Y+"),
    observable_labels: tuple[str, ...] | None = None,
) -> PrepareMeasureProtocol:
    """Return a configurable product-state candidate bank.

    The default bank has 16 inputs and all 15 nonidentity Pauli observables,
    providing 240 candidates from which an identifiable 64--96 feature bank can
    be selected.
    """

    input_labels = tuple(product(local_input_labels, repeat=2))
    input_states = tuple(product_state(labels) for labels in input_labels)
    if observable_labels is None:
        observable_labels = tuple(
            left + right
            for left, right in product(("I", "X", "Y", "Z"), repeat=2)
            if left + right != "II"
        )
    observables = tuple(pauli_word(label) for label in observable_labels)
    settings = tuple(product(range(len(input_states)), range(len(observables))))
    return PrepareMeasureProtocol(
        name="two_qubit_product_candidate_bank",
        n_qubits=2,
        input_labels=input_labels,
        input_states=input_states,
        observable_labels=observable_labels,
        observables=observables,
        settings=settings,
    )


@lru_cache(maxsize=16)
def _cached_two_qubit_protocol(
    gate_name: str, basis: tuple[str, ...], target_features: int
) -> PrepareMeasureProtocol:
    from .identifiability import representative_parameters, select_protocol

    parameters = representative_parameters(gate_name=gate_name, basis=basis)
    selected, _ = select_protocol(
        parameters,
        two_qubit_candidate_protocol(),
        target_features=target_features,
    )
    return selected


def two_qubit_protocol(
    gate_name: str = "CX",
    basis: tuple[str, ...] = ("ZI", "IZ", "ZX", "ZZ"),
    target_features: int = 80,
) -> PrepareMeasureProtocol:
    """Automatically construct an identifiable 64--96 feature probe bank."""

    if not 64 <= target_features <= 96:
        raise ValueError("target_features must lie between 64 and 96")
    return _cached_two_qubit_protocol(
        gate_name.upper(), tuple(label.upper() for label in basis), target_features
    )


def _readout_observable(
    observable_label: str,
    confusion: ReadoutConfusion,
) -> np.ndarray:
    """Return the effective observable after independent classical confusion."""

    n_qubits = len(observable_label)
    offset, scale = confusion.coefficients(n_qubits)
    factors: list[np.ndarray] = []
    for qubit, letter in enumerate(observable_label):
        if letter == "I":
            factors.append(I)
        else:
            factors.append(offset[qubit] * I + scale[qubit] * pauli_word(letter))
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def prepare_measure_expectations(
    channel: QuantumChannel,
    protocol: PrepareMeasureProtocol,
    readout_confusion: ReadoutConfusion | None = None,
) -> np.ndarray:
    """Calculate exact physical or readout-corrupted expectation values."""

    if channel.n_qubits != protocol.n_qubits:
        raise ValueError("Channel and protocol act on different numbers of qubits")
    outputs: dict[int, np.ndarray] = {}
    effective_observables: dict[int, np.ndarray] = {}
    features = np.empty(protocol.feature_count, dtype=float)
    for feature_index, (input_index, observable_index) in enumerate(protocol.settings):
        if input_index not in outputs:
            outputs[input_index] = channel.apply(protocol.input_states[input_index])
        if observable_index not in effective_observables:
            if readout_confusion is None:
                effective_observables[observable_index] = protocol.observables[
                    observable_index
                ]
            else:
                effective_observables[observable_index] = _readout_observable(
                    protocol.observable_labels[observable_index],
                    readout_confusion,
                )
        value = np.trace(effective_observables[observable_index] @ outputs[input_index])
        features[feature_index] = float(np.real_if_close(value))
    return np.clip(features, -1.0, 1.0)


def expectation_from_choi(
    channel: QuantumChannel, input_state: np.ndarray, observable: np.ndarray
) -> float:
    """Evaluate ``Tr[O E(rho)]`` from the normalized Choi state."""

    choi_state = channel_to_choi(channel, normalized=True)
    value = channel.dimension * np.trace(
        np.kron(np.asarray(input_state).T, np.asarray(observable)) @ choi_state
    )
    return float(np.real_if_close(value))


def sample_expectations(
    expectations: np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert exact two-outcome expectations into binomial shot estimates."""

    if shots <= 0:
        raise ValueError("shots must be a positive integer")
    values = np.clip(np.asarray(expectations, dtype=float), -1.0, 1.0)
    plus_probability = (1.0 + values) / 2.0
    plus_counts = rng.binomial(shots, plus_probability)
    return 2.0 * plus_counts / shots - 1.0


def extract_features(
    channel: QuantumChannel,
    protocol: PrepareMeasureProtocol | ChoiStateProtocol | str,
    shots: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    readout_confusion: ReadoutConfusion | None = None,
) -> np.ndarray:
    """Extract protocol features with optional finite-shot sampling."""

    if isinstance(protocol, str):
        normalized = protocol.lower()
        if normalized in {"choi", "choi_state", "protocol_a"}:
            protocol = ChoiStateProtocol()
        elif normalized in {"prepare_measure", "protocol_b"}:
            protocol = (
                one_qubit_protocol()
                if channel.n_qubits == 1
                else two_qubit_protocol(
                    gate_name=channel.metadata.get("gate_name", "CX"),
                    basis=tuple(
                        channel.metadata.get(
                            "coherent_labels", ("ZI", "IZ", "ZX", "ZZ")
                        )
                    ),
                )
            )
        else:
            raise ValueError(f"Unknown protocol: {protocol!r}")

    if isinstance(protocol, ChoiStateProtocol):
        if shots is not None or readout_confusion is not None:
            raise ValueError("The Choi reference protocol is exact-only")
        choi = channel_to_choi(channel, normalized=True)
        return np.concatenate((choi.real.reshape(-1), choi.imag.reshape(-1)))

    expectations = prepare_measure_expectations(
        channel, protocol, readout_confusion=readout_confusion
    )
    if shots is None:
        return expectations
    return sample_expectations(
        expectations,
        int(shots),
        np.random.default_rng() if rng is None else rng,
    )
