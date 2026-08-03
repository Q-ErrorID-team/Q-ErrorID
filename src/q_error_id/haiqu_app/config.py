"""Configuration and supported Haiqu mitigation modes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MitigationMode(str, Enum):
    """Stable names for the six requested Haiqu comparisons."""

    RAW = "raw"
    DEFAULT = "default_mitigation"
    READOUT = "readout_only"
    DYNAMICAL_DECOUPLING = "dynamical_decoupling"
    NOISE_TAILORING = "noise_tailoring"
    ADVANCED = "advanced_combined"

    @property
    def use_mitigation(self) -> bool:
        return self is not MitigationMode.RAW

    @property
    def error_mitigation_options(self) -> dict[str, bool]:
        """Return the exact option names supported by haiqu-sdk 1.3.1."""

        if self is MitigationMode.RAW:
            return {}
        if self is MitigationMode.DEFAULT:
            return {}
        if self is MitigationMode.READOUT:
            return {
                "dynamical_decoupling": False,
                "readout_mitigation": True,
                "noise_tailoring": False,
                "advanced_mitigation": False,
            }
        if self is MitigationMode.DYNAMICAL_DECOUPLING:
            return {
                "dynamical_decoupling": True,
                "readout_mitigation": False,
                "noise_tailoring": False,
                "advanced_mitigation": False,
            }
        if self is MitigationMode.NOISE_TAILORING:
            return {
                "dynamical_decoupling": False,
                "readout_mitigation": False,
                "noise_tailoring": True,
                "advanced_mitigation": False,
            }
        return {
            "dynamical_decoupling": True,
            "readout_mitigation": True,
            "noise_tailoring": True,
            "advanced_mitigation": True,
        }

    @property
    def run_options(self) -> dict:
        options = self.error_mitigation_options
        return {"error_mitigation_options": options} if options else {}


@dataclass(slots=True)
class ExecutionConfig:
    """Runtime configuration with secrets sourced only from the environment."""

    device: str = "fake_fez"
    shots: int = 4096
    seed: int = 2026
    require_cloud: bool = False
    allow_local_fallback: bool = True
    model_mode: str = "compare"
    optimization_level: int = 2
    demo_qubit_count: int = 4
    validation_repeats: int = 1
    evaluation_repeats: int = 1
    benchmark_two_qubit_depths: tuple[int, ...] = (2, 4, 8, 16)
    readout_regularization: float = 1e-6
    response_regularization: float = 3e-2
    verbose: bool = False
    output_root: Path = Path(".")
    model_root: Path = Path("artifacts/models")
    data_root: Path = Path("artifacts/datasets")

    @classmethod
    def from_env(cls, **overrides) -> ExecutionConfig:
        output_root = Path(overrides.get("output_root", ".")).resolve()
        values = {
            "model_mode": os.environ.get("Q_ERROR_ID_MODEL_MODE", "compare"),
            "model_root": Path(
                os.environ.get(
                    "Q_ERROR_ID_MODEL_ROOT",
                    str(output_root / "artifacts" / "models"),
                )
            ),
            "data_root": Path(
                os.environ.get(
                    "Q_ERROR_ID_DATA_ROOT",
                    str(output_root / "artifacts" / "datasets"),
                )
            ),
        }
        values.update(overrides)
        return cls(**values)

    @property
    def haiqu_api_key(self) -> str | None:
        return os.environ.get("HAIQU_API_KEY") or None

    @property
    def ibm_credentials(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if token := os.environ.get("IBM_QUANTUM_TOKEN"):
            result["ibm_quantum_token"] = token
        if instance := os.environ.get("IBM_QUANTUM_INSTANCE"):
            result["ibm_quantum_instance"] = instance
        return result

    def validate(self) -> None:
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.optimization_level not in {0, 1, 2, 3}:
            raise ValueError("optimization_level must be 0, 1, 2, or 3")
        if self.demo_qubit_count < 2:
            raise ValueError("demo_qubit_count must be at least 2")
        if self.validation_repeats < 1 or self.evaluation_repeats < 1:
            raise ValueError("validation_repeats and evaluation_repeats must be positive")
        depths = tuple(int(depth) for depth in self.benchmark_two_qubit_depths)
        if not depths or any(depth < 2 or depth % 2 for depth in depths):
            raise ValueError(
                "benchmark_two_qubit_depths must contain even integers >= 2"
            )
        if len(set(depths)) != len(depths):
            raise ValueError("benchmark_two_qubit_depths must not contain duplicates")
        self.benchmark_two_qubit_depths = depths
        if self.readout_regularization < 0.0:
            raise ValueError("readout_regularization must be nonnegative")
        if self.response_regularization < 0.0:
            raise ValueError("response_regularization must be nonnegative")
        self.model_mode = self.model_mode.lower()
        if self.model_mode not in {"ridge", "qnn", "compare"}:
            raise ValueError("model_mode must be 'ridge', 'qnn', or 'compare'")
        if self.require_cloud and not self.haiqu_api_key:
            raise RuntimeError(
                "HAIQU_API_KEY is required because cloud execution was requested"
            )


EXPERIMENT_GROUPS = {
    "root": "Q-ErrorID Hackathon",
    "diagnostics": "Q-ErrorID / diagnostics",
    "model": "Q-ErrorID / model deployment",
    "mitigation": "Q-ErrorID / mitigation benchmark",
    "final": "Q-ErrorID / final demo",
}
