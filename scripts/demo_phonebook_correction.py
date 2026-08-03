#!/usr/bin/env python
"""Standalone demo: QROM 'phonebook' lookup on 5 qubits, all correction methods.

Independent of the Grover training/validation/depth-sweep loop. By default it
encodes the *full* 8-entry phonebook in superposition (the original notebook
circuit: Walsh-Hadamard on the index register, then one X-select/mcx-write/
X-unselect block per entry), not just a single deterministic lookup.

Five methods are compared:
  1. raw                    -- no correction
  2. readout_only           -- independent readout-assignment inverse
  3. learned                -- readout inverse + generator response-matrix
                                inverse, gated per index by validation repeats
  4. haiqu_mitigation_only  -- Haiqu's own mitigation stack (cloud only)
  5. learned_plus_haiqu     -- generator correction on top of Haiqu's
                                mitigated distribution, gated per index
                                (cloud only)

Methods 4-5 require a real Haiqu Cloud session (--require-cloud with
HAIQU_API_KEY set); without a key they are reported as
"not_run_requires_haiqu_cloud".

Validation gate: for each of the 8 phonebook indices independently, the
generator correction is only enabled if held-out validation repeats show a
positive mean paired improvement in "probability of the correct data value"
over the readout-only (or Haiqu-mitigated) baseline, with bounded
quasiprobability negativity/simplex-projection cost -- mirroring the
edge-adaptive validation gate used in scripts/run_end_to_end_demo.py. An
index that fails validation keeps its baseline (never gets worse from our
own correction) for the held-out evaluation repeats used in the final report.

Usage:
    python scripts/demo_phonebook_correction.py --shots 4096
    python scripts/demo_phonebook_correction.py --mode lookup --index 101
    python scripts/demo_phonebook_correction.py --shots 4096 --require-cloud \
        --validation-repeats 2 --evaluation-repeats 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.transpiler import CouplingMap

from q_error_id.haiqu_app.backend import HaiquSession
from q_error_id.haiqu_app.circuits import (
    build_one_qubit_diagnostics,
    build_readout_calibration_circuits,
    build_two_qubit_diagnostics,
)
from q_error_id.haiqu_app.config import ExecutionConfig, MitigationMode
from q_error_id.haiqu_app.models import ModelRepository
from q_error_id.haiqu_app.phonebook_demo import (
    PHONEBOOK,
    PhonebookResponseModel,
    build_phonebook_lookup_circuit,
    build_phonebook_superposition_circuit,
    conditional_data_distributions,
    ideal_joint_distribution,
    joint_distribution_from_conditionals,
)
from q_error_id.haiqu_app.readout import ReadoutCalibrationBundle
from q_error_id.haiqu_app.results import results_to_features, total_variation_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="fake_fez")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--optimization-level", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=("superposition", "lookup"),
        default="superposition",
        help="Encode all 8 entries in superposition, or a single deterministic lookup.",
    )
    parser.add_argument(
        "--index", default="101", help="3-bit index for --mode lookup (000-111)."
    )
    parser.add_argument("--response-regularization", type=float, default=0.03)
    parser.add_argument(
        "--validation-repeats",
        type=int,
        default=2,
        help="Independent repeats reserved for the per-index correction gate.",
    )
    parser.add_argument(
        "--evaluation-repeats",
        type=int,
        default=2,
        help="Held-out repeats used for the final reported distributions.",
    )
    parser.add_argument(
        "--require-cloud",
        action="store_true",
        help="Fail instead of local fallback; required for the two Haiqu-mitigation methods.",
    )
    parser.add_argument("--no-local-fallback", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    return parser.parse_args()


def _compact_coupling(qubits, edges) -> CouplingMap:
    mapping = {int(q): i for i, q in enumerate(qubits)}
    directed = []
    for left, right in edges:
        a, b = mapping[int(left)], mapping[int(right)]
        directed.extend([(a, b), (b, a)])
    return CouplingMap(directed)


def _execute_raw(
    session: HaiquSession, circuits, *, group: str, job_name: str, qubits, edges,
    seed: int | None = None,
):
    """Run circuits either through Haiqu Cloud (raw) or the local honest fallback."""

    if session.cloud_enabled:
        logged = session.log_circuits(circuits, group=group)
        execution_circuits, _analytics = session.transpile_cloud(logged, group=group)
        results, _job = session.run_cloud(
            execution_circuits, mode=MitigationMode.RAW, group=group, job_name=job_name
        )
        return execution_circuits, results
    execution_circuits, results, _analytics = session.local_transpile_and_run(
        circuits, physical_qubits=qubits, tree_edges=edges, seed=seed
    )
    return execution_circuits, results


def _execute_haiqu_mitigated(session: HaiquSession, circuits, *, group: str, job_name: str):
    if not session.cloud_enabled:
        raise RuntimeError("Haiqu-mitigated execution requires an active cloud session")
    logged = session.log_circuits(circuits, group=group)
    execution_circuits, _analytics = session.transpile_cloud(logged, group=group)
    results, _job = session.run_cloud(
        execution_circuits, mode=MitigationMode.ADVANCED, group=group, job_name=job_name
    )
    return results


def _collect_repeats(
    session, circuit_builder, *, index_qubits, data_qubits, qubits, edges,
    repeats: int, seed_base: int, group: str, job_prefix: str, use_haiqu_mitigation: bool,
):
    samples = []
    for i in range(repeats):
        circuit = circuit_builder()
        if use_haiqu_mitigation:
            results = _execute_haiqu_mitigated(
                session, [circuit], group=group, job_name=f"{job_prefix} {i}"
            )
        else:
            _exec, results = _execute_raw(
                session, [circuit], group=group, job_name=f"{job_prefix} {i}",
                qubits=qubits, edges=edges, seed=seed_base + i,
            )
        counts = results[0]
        conditional, index_marginal = conditional_data_distributions(
            counts, index_qubits=index_qubits, data_qubits=data_qubits
        )
        samples.append((conditional, index_marginal))
    return samples


def _validate_generator_gate(
    samples, *, response_models: dict[str, PhonebookResponseModel],
    readout_bundle: ReadoutCalibrationBundle | None, data_qubits_physical, apply_readout: bool,
) -> tuple[dict[str, bool], pd.DataFrame]:
    """Decide, per phonebook index, whether the generator correction reliably helps."""

    rows = []
    per_index: dict[str, dict[str, list[float]]] = {
        idx: {"improvement": [], "negativity": [], "simplex": []} for idx in response_models
    }
    for repeat_index, (conditional, _index_marginal) in enumerate(samples):
        for index_bits, response_model in response_models.items():
            distribution = conditional.get(index_bits, {})
            if sum(distribution.values()) <= 0.0:
                continue
            correct_answer = PHONEBOOK[int(index_bits, 2)]
            baseline = distribution
            if apply_readout:
                assignment = readout_bundle.independent_assignment(data_qubits_physical)
                baseline, _audit = assignment.correct_with_audit(baseline)
            baseline_success = baseline.get(correct_answer, 0.0)
            corrected, audit = response_model.correct_with_audit(baseline)
            corrected_success = corrected.get(correct_answer, 0.0)
            improvement = corrected_success - baseline_success
            per_index[index_bits]["improvement"].append(improvement)
            per_index[index_bits]["negativity"].append(audit["inverse_raw_negativity"])
            per_index[index_bits]["simplex"].append(audit["simplex_projection_l1"])
            rows.append(
                {
                    "repeat": repeat_index,
                    "index_bits": index_bits,
                    "correct_answer": correct_answer,
                    "baseline_success": baseline_success,
                    "corrected_success": corrected_success,
                    "improvement": improvement,
                    **audit,
                }
            )

    generator_enabled: dict[str, bool] = {}
    for index_bits, stats in per_index.items():
        improvements = stats["improvement"]
        if not improvements:
            generator_enabled[index_bits] = False
            continue
        mean_improvement = float(np.mean(improvements))
        positive_fraction = float(np.mean([v > 0 for v in improvements]))
        mean_negativity = float(np.mean(stats["negativity"]))
        mean_simplex = float(np.mean(stats["simplex"]))
        generator_enabled[index_bits] = bool(
            mean_improvement > 0
            and positive_fraction >= 0.5
            and mean_negativity <= 0.05
            and mean_simplex <= 0.10
        )
    return generator_enabled, pd.DataFrame(rows)


def _apply_gated_correction(
    samples, *, generator_enabled: dict[str, bool],
    response_models: dict[str, PhonebookResponseModel],
    readout_bundle: ReadoutCalibrationBundle | None, data_qubits_physical, apply_readout: bool,
) -> list[dict[str, float]]:
    """Apply the validated per-index gate to each evaluation repeat; return joint distributions."""

    joints = []
    for conditional, index_marginal in samples:
        corrected_conditional: dict[str, dict[str, float]] = {}
        for index_bits, distribution in conditional.items():
            current = distribution
            if sum(distribution.values()) <= 0.0:
                corrected_conditional[index_bits] = current
                continue
            if apply_readout:
                assignment = readout_bundle.independent_assignment(data_qubits_physical)
                current, _audit = assignment.correct_with_audit(current)
            if generator_enabled.get(index_bits, False):
                current = response_models[index_bits].correct(current)
            corrected_conditional[index_bits] = current
        joints.append(joint_distribution_from_conditionals(corrected_conditional, index_marginal))
    return joints


def _mean_distribution(joints: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for joint in joints for key in joint})
    return {key: float(np.mean([joint.get(key, 0.0) for joint in joints])) for key in keys}


def main() -> int:
    args = parse_args()
    if args.mode == "lookup" and (
        len(args.index) != 3 or any(bit not in "01" for bit in args.index)
    ):
        raise ValueError("--index must be a 3-bit binary string, e.g. 101")

    config = ExecutionConfig.from_env(
        device=args.device, shots=args.shots, seed=args.seed,
        optimization_level=args.optimization_level, demo_qubit_count=5,
        response_regularization=args.response_regularization,
        require_cloud=args.require_cloud, allow_local_fallback=not args.no_local_fallback,
        verbose=True, output_root=args.output_root,
    )
    session = HaiquSession(config)
    session.authenticate()
    selected = session.select_device(config.device)
    qubits, edges = session.subgraph()
    physical_to_compact = {physical: i for i, physical in enumerate(qubits)}
    index_qubits_physical = tuple(qubits[:3])
    data_qubits_physical = tuple(qubits[3:5])
    index_qubits = tuple(physical_to_compact[q] for q in index_qubits_physical)
    data_qubits = tuple(physical_to_compact[q] for q in data_qubits_physical)
    print(
        f"device={selected.id} cloud_enabled={session.cloud_enabled} "
        f"qubits={qubits} edges={edges}",
        file=sys.stderr,
    )

    # ---- 1. Diagnostics ----
    diagnostic_circuits = []
    for physical in qubits:
        diagnostic_circuits.extend(
            build_one_qubit_diagnostics(
                physical, gate_name="id", width=len(qubits),
                circuit_qubit=physical_to_compact[physical],
            )
        )
    edge_qubits = {}
    for physical_edge in edges:
        key = f"q{physical_edge[0]}-q{physical_edge[1]}"
        edge_qubits[key] = physical_edge
        diagnostic_circuits.extend(
            build_two_qubit_diagnostics(
                physical_edge, gate_name="cx", width=len(qubits),
                circuit_qubits=(
                    physical_to_compact[physical_edge[0]],
                    physical_to_compact[physical_edge[1]],
                ),
            )
        )
    _exec_diag, raw_diag_results = _execute_raw(
        session, diagnostic_circuits, group="diagnostics",
        job_name="Q-ErrorID phonebook demo diagnostics", qubits=qubits, edges=edges,
        seed=args.seed,
    )

    # ---- 2. Readout calibration ----
    calibration_circuits = build_readout_calibration_circuits(qubits, edges, width=len(qubits))
    _exec_cal, calibration_results = _execute_raw(
        session, calibration_circuits, group="diagnostics",
        job_name="Q-ErrorID phonebook demo readout calibration", qubits=qubits, edges=edges,
        seed=args.seed,
    )
    readout_bundle = ReadoutCalibrationBundle.from_results(
        calibration_circuits, calibration_results, expected_shots=args.shots
    )
    print(
        f"readout calibration validation_passed={readout_bundle.validation_passed}",
        file=sys.stderr,
    )
    diag_features_corrected = results_to_features(
        diagnostic_circuits, raw_diag_results, mode="readout_corrected",
        readout_calibration=readout_bundle,
    )

    # ---- 3. Reconstruct local channels (Ridge) and per-index response models ----
    models = ModelRepository(
        args.output_root / "artifacts" / "models",
        args.output_root / "artifacts" / "datasets",
        shots=args.shots,
    )
    single_qubit_channels: dict[int, dict] = {}
    for physical in qubits:
        estimate = models.predict(
            "1q", diag_features_corrected.features[f"q{physical}"], model_kind="ridge"
        )
        single_qubit_channels[physical_to_compact[physical]] = estimate.to_dict()
    two_qubit_channels: dict[tuple[int, int], dict] = {}
    for key, physical_edge in edge_qubits.items():
        estimate = models.predict(
            "2q", diag_features_corrected.features[key], model_kind="ridge"
        )
        compact_edge = (
            physical_to_compact[physical_edge[0]], physical_to_compact[physical_edge[1]]
        )
        two_qubit_channels[compact_edge] = estimate.to_dict()

    compact_coupling = _compact_coupling(qubits, edges)
    index_width = len(index_qubits)
    response_models: dict[str, PhonebookResponseModel] = {}
    for i in range(2**index_width):
        index_bits = format(i, f"0{index_width}b")
        response_models[index_bits] = PhonebookResponseModel.from_channels(
            index_bits, index_qubits=index_qubits, data_qubits=data_qubits,
            width=len(qubits), single_qubit_channels=single_qubit_channels,
            two_qubit_channels=two_qubit_channels, coupling_map=compact_coupling,
            regularization=args.response_regularization,
        )
    print(
        "response condition numbers: "
        + json.dumps({k: round(v.condition_number, 3) for k, v in response_models.items()}),
        file=sys.stderr,
    )

    # ---- 4. Build the target circuit builder ----
    if args.mode == "superposition":
        def circuit_builder():
            return build_phonebook_superposition_circuit(
                index_qubits=index_qubits, data_qubits=data_qubits, width=len(qubits)
            )
        ideal_joint = ideal_joint_distribution()
    else:
        true_data = PHONEBOOK[int(args.index, 2)]
        def circuit_builder():
            return build_phonebook_lookup_circuit(
                args.index, true_data, index_qubits=index_qubits, data_qubits=data_qubits,
                width=len(qubits),
            )
        ideal_joint = {f"{args.index}{true_data}": 1.0}

    # ---- 5. Validation gate: "learned" (readout + generator on raw execution) ----
    validation_raw = _collect_repeats(
        session, circuit_builder, index_qubits=index_qubits, data_qubits=data_qubits,
        qubits=qubits, edges=edges, repeats=args.validation_repeats,
        seed_base=args.seed + 10_000, group="diagnostics",
        job_prefix="Q-ErrorID phonebook validation raw", use_haiqu_mitigation=False,
    )
    generator_enabled_learned, validation_table_learned = _validate_generator_gate(
        validation_raw, response_models=response_models, readout_bundle=readout_bundle,
        data_qubits_physical=data_qubits_physical, apply_readout=True,
    )
    print(f"learned generator_enabled per index: {generator_enabled_learned}", file=sys.stderr)

    # ---- 6. Held-out evaluation: raw / readout_only / learned ----
    evaluation_raw = _collect_repeats(
        session, circuit_builder, index_qubits=index_qubits, data_qubits=data_qubits,
        qubits=qubits, edges=edges, repeats=args.evaluation_repeats,
        seed_base=args.seed + 20_000, group="final",
        job_prefix="Q-ErrorID phonebook evaluation raw", use_haiqu_mitigation=False,
    )
    raw_joints = [
        joint_distribution_from_conditionals(conditional, index_marginal)
        for conditional, index_marginal in evaluation_raw
    ]
    readout_only_joints = _apply_gated_correction(
        evaluation_raw, generator_enabled={idx: False for idx in response_models},
        response_models=response_models, readout_bundle=readout_bundle,
        data_qubits_physical=data_qubits_physical, apply_readout=True,
    )
    learned_joints = _apply_gated_correction(
        evaluation_raw, generator_enabled=generator_enabled_learned,
        response_models=response_models, readout_bundle=readout_bundle,
        data_qubits_physical=data_qubits_physical, apply_readout=True,
    )

    distributions: dict[str, dict[str, float]] = {
        "ideal": ideal_joint,
        "raw": _mean_distribution(raw_joints),
        "readout_only": _mean_distribution(readout_only_joints),
        "learned": _mean_distribution(learned_joints),
    }
    statuses = {name: "executed" for name in distributions}

    # ---- 7. Haiqu-mitigation methods (cloud only), also validation-gated ----
    if session.cloud_enabled:
        validation_haiqu = _collect_repeats(
            session, circuit_builder, index_qubits=index_qubits, data_qubits=data_qubits,
            qubits=qubits, edges=edges, repeats=args.validation_repeats,
            seed_base=args.seed + 30_000, group="mitigation",
            job_prefix="Q-ErrorID phonebook validation haiqu", use_haiqu_mitigation=True,
        )
        generator_enabled_haiqu, validation_table_haiqu = _validate_generator_gate(
            validation_haiqu, response_models=response_models, readout_bundle=None,
            data_qubits_physical=data_qubits_physical, apply_readout=False,
        )
        print(f"learned+haiqu generator_enabled per index: {generator_enabled_haiqu}", file=sys.stderr)

        evaluation_haiqu = _collect_repeats(
            session, circuit_builder, index_qubits=index_qubits, data_qubits=data_qubits,
            qubits=qubits, edges=edges, repeats=args.evaluation_repeats,
            seed_base=args.seed + 40_000, group="final",
            job_prefix="Q-ErrorID phonebook evaluation haiqu", use_haiqu_mitigation=True,
        )
        haiqu_only_joints = [
            joint_distribution_from_conditionals(conditional, index_marginal)
            for conditional, index_marginal in evaluation_haiqu
        ]
        learned_plus_haiqu_joints = _apply_gated_correction(
            evaluation_haiqu, generator_enabled=generator_enabled_haiqu,
            response_models=response_models, readout_bundle=None,
            data_qubits_physical=data_qubits_physical, apply_readout=False,
        )
        distributions["haiqu_mitigation_only"] = _mean_distribution(haiqu_only_joints)
        distributions["learned_plus_haiqu"] = _mean_distribution(learned_plus_haiqu_joints)
        statuses["haiqu_mitigation_only"] = "executed_haiqu_cloud"
        statuses["learned_plus_haiqu"] = "executed_haiqu_cloud"

        results_dir = args.output_root / "results" / "haiqu"
        results_dir.mkdir(parents=True, exist_ok=True)
        validation_table_learned["stack"] = "learned"
        validation_table_haiqu["stack"] = "learned_plus_haiqu"
        pd.concat([validation_table_learned, validation_table_haiqu]).to_csv(
            results_dir / "phonebook_correction_validation.csv", index=False
        )
    else:
        distributions["haiqu_mitigation_only"] = {}
        distributions["learned_plus_haiqu"] = {}
        statuses["haiqu_mitigation_only"] = "not_run_requires_haiqu_cloud"
        statuses["learned_plus_haiqu"] = "not_run_requires_haiqu_cloud"
        results_dir = args.output_root / "results" / "haiqu"
        results_dir.mkdir(parents=True, exist_ok=True)
        validation_table_learned["stack"] = "learned"
        validation_table_learned.to_csv(
            results_dir / "phonebook_correction_validation.csv", index=False
        )

    # ---- 8. Report ----
    rows = []
    for name, distribution in distributions.items():
        if name == "ideal" or not distribution:
            tvd = 0.0 if name == "ideal" else None
            success = 1.0 if name == "ideal" else None
        else:
            tvd = total_variation_distance(distribution, ideal_joint)
            success = sum(distribution.get(k, 0.0) for k in ideal_joint)
        rows.append(
            {
                "method": name,
                "status": statuses[name],
                "tvd_to_ideal": tvd,
                "p_correct_joint": success,
                "distribution": json.dumps(distribution),
            }
        )
    table = pd.DataFrame(rows)
    print(table[["method", "status", "tvd_to_ideal", "p_correct_joint"]].to_string(index=False))

    results_dir = args.output_root / "results" / "haiqu"
    suffix = args.mode if args.mode == "superposition" else f"lookup_idx{args.index}"
    table.to_csv(results_dir / f"phonebook_demo_{suffix}.csv", index=False)

    # ---- 9. Bar charts: one per method, correct bitstrings highlighted ----
    all_outcomes = sorted(set(ideal_joint) | {k for d in distributions.values() for k in d})
    correct_outcomes = set(ideal_joint)
    plot_methods = [name for name in distributions if distributions[name] or name == "ideal"]
    fig, axes = plt.subplots(
        len(plot_methods), 1, figsize=(max(10, 0.35 * len(all_outcomes)), 3 * len(plot_methods)),
        squeeze=False, constrained_layout=True,
    )
    for row, name in enumerate(plot_methods):
        axis = axes[row][0]
        distribution = distributions[name]
        values = [distribution.get(outcome, 0.0) for outcome in all_outcomes]
        colors = [
            "tab:green" if outcome in correct_outcomes else "tab:gray"
            for outcome in all_outcomes
        ]
        axis.bar(all_outcomes, values, color=colors)
        p_correct = sum(distribution.get(o, 0.0) for o in correct_outcomes)
        p_incorrect = sum(
            distribution.get(o, 0.0) for o in all_outcomes if o not in correct_outcomes
        )
        axis.set_title(
            f"{name}  (status={statuses[name]})\n"
            f"P(correct)={p_correct:.3f}   P(incorrect)={p_incorrect:.3f}"
        )
        axis.set_ylabel("probability")
        axis.tick_params(axis="x", rotation=90, labelsize=7)
        axis.set_ylim(0, max(0.2, max(values, default=0.0) * 1.15))
    plot_path = results_dir / f"phonebook_demo_{suffix}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"wrote {plot_path}", file=sys.stderr)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "execution_mode": "haiqu_cloud" if session.cloud_enabled else "local_fallback",
                "physical_qubits": list(qubits),
                "index_qubits_physical": list(index_qubits_physical),
                "data_qubits_physical": list(data_qubits_physical),
                "correct_outcomes": sorted(correct_outcomes),
                "generator_enabled_learned": generator_enabled_learned,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
