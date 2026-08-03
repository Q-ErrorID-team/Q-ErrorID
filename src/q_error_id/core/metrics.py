"""Evaluation metrics for generator regression and physical channels."""

from __future__ import annotations

import numpy as np

from .channels import QuantumChannel
from .parameters import ChannelParameters
from .representations import (
    channel_to_choi,
    channel_to_ptm,
    partial_trace_choi_output,
)


def parameter_mae(target: np.ndarray, prediction: np.ndarray) -> float:
    """Mean absolute parameter error."""

    return float(np.mean(np.abs(np.asarray(target) - np.asarray(prediction))))


def parameter_rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Root-mean-square parameter error."""

    difference = np.asarray(target) - np.asarray(prediction)
    return float(np.sqrt(np.mean(np.abs(difference) ** 2)))


def relative_ptm_frobenius_error(
    target: QuantumChannel | np.ndarray,
    prediction: QuantumChannel | np.ndarray,
) -> float:
    """Relative Frobenius error between Pauli transfer matrices."""

    target_ptm = (
        channel_to_ptm(target) if isinstance(target, QuantumChannel) else target
    )
    predicted_ptm = (
        channel_to_ptm(prediction)
        if isinstance(prediction, QuantumChannel)
        else prediction
    )
    denominator = np.linalg.norm(target_ptm, ord="fro")
    numerator = np.linalg.norm(predicted_ptm - target_ptm, ord="fro")
    return float(numerator / max(denominator, np.finfo(float).eps))


def _positive_sqrt(matrix: np.ndarray) -> np.ndarray:
    hermitian = (matrix + matrix.conj().T) / 2.0
    values, vectors = np.linalg.eigh(hermitian)
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.conj().T


def choi_state_fidelity(
    target: QuantumChannel | np.ndarray,
    prediction: QuantumChannel | np.ndarray,
) -> float:
    """Uhlmann fidelity between normalized Choi states."""

    target_state = (
        channel_to_choi(target, normalized=True)
        if isinstance(target, QuantumChannel)
        else np.asarray(target, dtype=np.complex128)
    )
    predicted_state = (
        channel_to_choi(prediction, normalized=True)
        if isinstance(prediction, QuantumChannel)
        else np.asarray(prediction, dtype=np.complex128)
    )
    target_state = target_state / np.trace(target_state)
    predicted_state = predicted_state / np.trace(predicted_state)
    root = _positive_sqrt(target_state)
    sandwich = root @ predicted_state @ root
    fidelity = float(np.real(np.trace(_positive_sqrt(sandwich))) ** 2)
    return float(np.clip(fidelity, 0.0, 1.0))


def heldout_observable_prediction_error(
    target_expectations: np.ndarray,
    predicted_expectations: np.ndarray,
) -> float:
    """RMSE on observables excluded from model fitting."""

    return parameter_rmse(target_expectations, predicted_expectations)


def trace_preservation_violation(channel: QuantumChannel) -> float:
    """Frobenius norm of ``Tr_output(J) - I`` for the unnormalized Choi matrix."""

    choi = channel_to_choi(channel, normalized=False)
    residual = partial_trace_choi_output(choi, channel.dimension) - np.eye(
        channel.dimension
    )
    return float(np.linalg.norm(residual, ord="fro"))


def minimum_choi_eigenvalue(channel: QuantumChannel) -> float:
    """Smallest eigenvalue of the normalized Choi state."""

    choi = channel_to_choi(channel, normalized=True)
    return float(np.linalg.eigvalsh((choi + choi.conj().T) / 2.0).min())


def error_strengths(parameters: ChannelParameters) -> dict[str, float]:
    """Summarize coherent and incoherent generator strengths."""

    incoherent = float(np.sum(parameters.gamma))
    if parameters.kappa_down is not None:
        incoherent += float(np.sum(parameters.kappa_down))
    return {
        "coherent_l2": float(np.linalg.norm(parameters.alpha)),
        "incoherent_l1": incoherent,
    }


def unitarity(channel: QuantumChannel) -> float:
    """Return channel unitarity from the traceless PTM block.

    This is valid for trace-preserving channels and does not use a unitary-only
    process-fidelity expression.
    """

    ptm = channel_to_ptm(channel)
    traceless_block = ptm[1:, 1:]
    value = np.trace(traceless_block.T @ traceless_block) / (channel.dimension**2 - 1)
    return float(np.clip(np.real(value), 0.0, 1.0))
