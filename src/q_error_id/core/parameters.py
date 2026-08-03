"""Shared parameter contracts for physical error generators."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

ONE_QUBIT_COHERENT = ("X", "Y", "Z")
ONE_QUBIT_STOCHASTIC = ("X", "Y", "Z")
GATE_BASES: dict[str, tuple[str, ...]] = {
    "CX": ("ZI", "IZ", "ZX", "ZZ"),
    "CZ": ("ZI", "IZ", "ZZ", "XX"),
    "RZZ": ("ZI", "IZ", "ZZ", "XX"),
}


@dataclass
class ChannelParameters:
    """Parameters of a local one- or two-qubit Lindblad generator.

    ``alpha`` and ``gamma`` correspond positionally to ``coherent_labels`` and
    ``stochastic_labels``.  For a one-qubit channel, ``kappa_down`` contains one
    amplitude-damping rate.  For a two-qubit gate-error channel, it may contain
    the two previously estimated local damping rates.
    """

    gate_name: str
    qubits: tuple[int, ...]
    coherent_labels: tuple[str, ...]
    stochastic_labels: tuple[str, ...]
    alpha: np.ndarray
    gamma: np.ndarray
    kappa_down: np.ndarray | None

    def __post_init__(self) -> None:
        self.qubits = tuple(int(q) for q in self.qubits)
        self.coherent_labels = tuple(label.upper() for label in self.coherent_labels)
        self.stochastic_labels = tuple(
            label.upper() for label in self.stochastic_labels
        )
        self.alpha = np.asarray(self.alpha, dtype=float).reshape(-1)
        self.gamma = np.asarray(self.gamma, dtype=float).reshape(-1)
        if self.kappa_down is not None:
            self.kappa_down = np.asarray(self.kappa_down, dtype=float).reshape(-1)
        self.validate()

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    def validate(self) -> None:
        """Validate dimensions, labels, and physical rate constraints."""

        if self.n_qubits not in (1, 2):
            raise ValueError("Only one- and two-qubit channels are supported")
        if len(self.coherent_labels) != self.alpha.size:
            raise ValueError("coherent_labels and alpha must have equal lengths")
        if len(self.stochastic_labels) != self.gamma.size:
            raise ValueError("stochastic_labels and gamma must have equal lengths")
        if np.any(self.gamma < 0.0):
            raise ValueError("Stochastic rates must be nonnegative")
        expected_label_length = self.n_qubits
        all_labels = self.coherent_labels + self.stochastic_labels
        if any(len(label) != expected_label_length for label in all_labels):
            raise ValueError("Every Pauli label must match the number of qubits")
        if self.kappa_down is not None:
            expected = 1 if self.n_qubits == 1 else 2
            if self.kappa_down.size != expected:
                raise ValueError(f"kappa_down must contain {expected} value(s)")
            if np.any(self.kappa_down < 0.0):
                raise ValueError("Amplitude-damping rates must be nonnegative")

    def as_vector(self, include_kappa: bool = True) -> np.ndarray:
        """Flatten parameters in the shared ``alpha, gamma, kappa`` order."""

        arrays = [self.alpha, self.gamma]
        if include_kappa and self.kappa_down is not None:
            arrays.append(self.kappa_down)
        return np.concatenate(arrays).astype(float, copy=True)

    def with_vector(
        self, values: np.ndarray, include_kappa: bool = True
    ) -> ChannelParameters:
        """Return a copy populated from :meth:`as_vector` ordering."""

        vector = np.asarray(values, dtype=float).reshape(-1)
        n_alpha = self.alpha.size
        n_gamma = self.gamma.size
        n_kappa = (
            self.kappa_down.size if include_kappa and self.kappa_down is not None else 0
        )
        if vector.size != n_alpha + n_gamma + n_kappa:
            raise ValueError("Parameter vector has the wrong length")
        alpha = vector[:n_alpha]
        gamma = vector[n_alpha : n_alpha + n_gamma]
        kappa = self.kappa_down
        if n_kappa:
            kappa = vector[-n_kappa:]
        return replace(self, alpha=alpha, gamma=gamma, kappa_down=kappa)

    def labels(self, include_kappa: bool = True) -> tuple[str, ...]:
        """Return stable, human-readable labels matching :meth:`as_vector`."""

        labels = tuple(f"alpha_{label}" for label in self.coherent_labels)
        labels += tuple(f"gamma_{label}" for label in self.stochastic_labels)
        if include_kappa and self.kappa_down is not None:
            if self.n_qubits == 1:
                labels += ("kappa_down",)
            else:
                labels += tuple(f"kappa_down_q{qubit}" for qubit in self.qubits)
        return labels


def one_qubit_parameters(
    alpha: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    kappa_down: float | np.ndarray = 0.0,
    qubit: int = 0,
) -> ChannelParameters:
    """Construct one-qubit parameters with the standard XYZ generator basis."""

    return ChannelParameters(
        gate_name="1Q",
        qubits=(qubit,),
        coherent_labels=ONE_QUBIT_COHERENT,
        stochastic_labels=ONE_QUBIT_STOCHASTIC,
        alpha=np.zeros(3) if alpha is None else alpha,
        gamma=np.zeros(3) if gamma is None else gamma,
        kappa_down=np.atleast_1d(kappa_down),
    )


def two_qubit_parameters(
    gate_name: str = "CX",
    alpha: np.ndarray | None = None,
    gamma: np.ndarray | None = None,
    kappa_down: np.ndarray | None = None,
    qubits: tuple[int, int] = (0, 1),
    basis: tuple[str, ...] | None = None,
) -> ChannelParameters:
    """Construct gate-specific parameters for CX, CZ, RZZ, or a custom basis."""

    normalized_name = gate_name.upper()
    labels = tuple(label.upper() for label in (basis or GATE_BASES[normalized_name]))
    if len(labels) != 4:
        raise ValueError("The gate-specific model requires four Pauli labels")
    return ChannelParameters(
        gate_name=normalized_name,
        qubits=qubits,
        coherent_labels=labels,
        stochastic_labels=labels,
        alpha=np.zeros(4) if alpha is None else alpha,
        gamma=np.zeros(4) if gamma is None else gamma,
        kappa_down=np.zeros(2) if kappa_down is None else kappa_down,
    )


def parameter_bounds(
    parameters: ChannelParameters, include_kappa: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return the project parameter bounds in flattened parameter order."""

    lower = [np.full(parameters.alpha.size, -0.15), np.zeros(parameters.gamma.size)]
    upper = [np.full(parameters.alpha.size, 0.15), np.full(parameters.gamma.size, 0.03)]
    if include_kappa and parameters.kappa_down is not None:
        lower.append(np.zeros(parameters.kappa_down.size))
        upper.append(np.full(parameters.kappa_down.size, 0.05))
    return np.concatenate(lower), np.concatenate(upper)
