"""Device-level aggregation and visualization of reconstructed generators."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .models import ChannelEstimate


def recommend_mitigation(channel: dict[str, Any]) -> str:
    """Map an interpretable generator to an operational recommendation."""

    coherent = float(channel.get("coherent_magnitude", 0.0))
    stochastic = float(channel.get("stochastic_magnitude", 0.0))
    damping = float(channel.get("amplitude_damping", 0.0))
    if damping > max(coherent, stochastic):
        return "shorten idle windows / remap; DD does not reverse amplitude damping"
    if coherent > 1.25 * stochastic:
        return "calibration/frame coherent correction, then validate"
    if stochastic > 1.25 * coherent:
        return "Haiqu readout/advanced mitigation; evaluate noise tailoring"
    return "combined coherent correction and Haiqu mitigation"


@dataclass
class DeviceErrorAtlas:
    """Serializable four-qubit local error atlas."""

    single_qubit_channels: dict[int, dict]
    two_qubit_channels: dict[str, dict]
    device_id: str
    calibration_timestamp: str
    physical_qubits: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_estimates(
        cls,
        *,
        device_id: str,
        physical_qubits: tuple[int, ...],
        single_qubit_estimates: dict[int, ChannelEstimate],
        two_qubit_estimates: dict[str, ChannelEstimate],
        edge_qubits: dict[str, tuple[int, int]],
        calibration_timestamp: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DeviceErrorAtlas:
        singles = {}
        for qubit, estimate in single_qubit_estimates.items():
            item = estimate.to_dict()
            item["physical_qubits"] = [int(qubit)]
            item["recommended_mitigation"] = recommend_mitigation(item)
            singles[int(qubit)] = item
        doubles = {}
        for edge, estimate in two_qubit_estimates.items():
            item = estimate.to_dict()
            item["physical_qubits"] = [int(q) for q in edge_qubits[edge]]
            item["recommended_mitigation"] = recommend_mitigation(item)
            doubles[edge] = item
        return cls(
            single_qubit_channels=singles,
            two_qubit_channels=doubles,
            device_id=device_id,
            calibration_timestamp=calibration_timestamp
            or datetime.now(timezone.utc).isoformat(),
            physical_qubits=tuple(int(q) for q in physical_qubits),
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "calibration_timestamp": self.calibration_timestamp,
            "physical_qubits": list(self.physical_qubits),
            "single_qubit_channels": {
                str(key): value for key, value in self.single_qubit_channels.items()
            },
            "two_qubit_channels": self.two_qubit_channels,
            "metadata": self.metadata,
        }

    def save_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return destination

    def dataframe(self) -> pd.DataFrame:
        rows = []
        for qubit, channel in sorted(self.single_qubit_channels.items()):
            rows.append(
                {
                    "kind": "node",
                    "channel": f"q{qubit}",
                    "physical_qubits": [qubit],
                    **{
                        key: channel[key]
                        for key in (
                            "coherent_magnitude",
                            "stochastic_magnitude",
                            "amplitude_damping",
                            "dominant_pauli_generator",
                            "recommended_mitigation",
                            "model_source",
                            "trained_model",
                        )
                    },
                }
            )
        for edge, channel in sorted(self.two_qubit_channels.items()):
            rows.append(
                {
                    "kind": "edge",
                    "channel": edge,
                    "physical_qubits": channel["physical_qubits"],
                    **{
                        key: channel[key]
                        for key in (
                            "coherent_magnitude",
                            "stochastic_magnitude",
                            "amplitude_damping",
                            "dominant_pauli_generator",
                            "recommended_mitigation",
                            "model_source",
                            "trained_model",
                        )
                    },
                }
            )
        return pd.DataFrame(rows)

    def plot(self, path: str | Path) -> Path:
        """Plot coherent/stochastic weights on the selected connected subgraph."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        qubits = self.physical_qubits or tuple(sorted(self.single_qubit_channels))
        positions = {
            qubit: (index, 0.18 * (-1) ** index) for index, qubit in enumerate(qubits)
        }
        metrics = (
            ("coherent_magnitude", "Coherent generator magnitude"),
            ("stochastic_magnitude", "Stochastic generator magnitude"),
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        all_values = [
            float(channel[metric])
            for metric, _ in metrics
            for channel in (
                list(self.single_qubit_channels.values())
                + list(self.two_qubit_channels.values())
            )
        ]
        vmax = max(all_values, default=1.0) or 1.0
        for axis, (metric, title) in zip(axes, metrics):
            for channel in self.two_qubit_channels.values():
                q0, q1 = channel["physical_qubits"]
                x = [positions[q0][0], positions[q1][0]]
                y = [positions[q0][1], positions[q1][1]]
                value = float(channel[metric])
                axis.plot(
                    x,
                    y,
                    color=plt.cm.magma(value / vmax),
                    linewidth=2.0 + 9.0 * value / vmax,
                    alpha=0.8,
                    zorder=1,
                )
                axis.text(
                    np.mean(x),
                    np.mean(y) + 0.09,
                    channel["dominant_pauli_generator"],
                    ha="center",
                    fontsize=8,
                )
            for qubit, channel in self.single_qubit_channels.items():
                x, y = positions[qubit]
                value = float(channel[metric])
                axis.scatter(
                    [x],
                    [y],
                    s=650,
                    c=[value],
                    cmap="magma",
                    vmin=0.0,
                    vmax=vmax,
                    edgecolor="black",
                    linewidth=1.0,
                    zorder=2,
                )
                axis.text(x, y, f"q{qubit}", ha="center", va="center", color="white")
                axis.text(
                    x,
                    y - 0.15,
                    f"κ↓={channel['amplitude_damping']:.3g}",
                    ha="center",
                    fontsize=8,
                )
            axis.set_title(title)
            axis.set_axis_off()
        fig.suptitle(
            f"Q-ErrorID local generator atlas — {self.device_id}\n"
            "Backend predictions are not ground truth; validate with held-out diagnostics",
            fontsize=12,
        )
        scalar = plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=0.0, vmax=vmax), cmap="magma"
        )
        fig.colorbar(
            scalar, ax=axes, fraction=0.025, pad=0.02, label="generator magnitude"
        )
        fig.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return destination
