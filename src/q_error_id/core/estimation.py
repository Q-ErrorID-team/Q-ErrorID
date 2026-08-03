"""Small reference optimizer used to validate exact parameter recovery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .channels import build_diagnostic_channel
from .parameters import ChannelParameters, parameter_bounds
from .protocols import PrepareMeasureProtocol, extract_features


@dataclass(frozen=True)
class RecoveryResult:
    """Result of bounded nonlinear least-squares recovery."""

    parameters: ChannelParameters
    success: bool
    cost: float
    residual_norm: float
    evaluations: int


def recover_parameters(
    template: ChannelParameters,
    protocol: PrepareMeasureProtocol,
    target_features: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    include_kappa: bool = True,
    max_evaluations: int = 1000,
) -> RecoveryResult:
    """Recover generator parameters from exact prepare-measure features."""

    lower, upper = parameter_bounds(template, include_kappa=include_kappa)
    if initial is None:
        initial = (lower + upper) / 2.0
        initial[: template.alpha.size] = 0.0
    initial = np.asarray(initial, dtype=float)
    target = np.asarray(target_features, dtype=float)

    def residual(vector: np.ndarray) -> np.ndarray:
        parameters = template.with_vector(vector, include_kappa=include_kappa)
        features = extract_features(build_diagnostic_channel(parameters), protocol)
        return features - target

    optimum = least_squares(
        residual,
        x0=initial,
        bounds=(lower, upper),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=max_evaluations,
    )
    recovered = template.with_vector(optimum.x, include_kappa=include_kappa)
    return RecoveryResult(
        parameters=recovered,
        success=bool(optimum.success),
        cost=float(optimum.cost),
        residual_norm=float(np.linalg.norm(optimum.fun)),
        evaluations=int(optimum.nfev),
    )
