# Q-ErrorID v0.61

Q-ErrorID reconstructs a local quantum-error generator instead of returning a
single gate-error number. It separates coherent Pauli terms, stochastic Pauli
rates, and one-qubit relaxation, then uses the reconstructed structure to
perform algorithm-level error cancellation and compare it with Haiqu
mitigation.

This corrected distribution is one installable project. It combines:

- physical one- and two-qubit channel models and diagnostic protocols;
- reproducible exact, readout-corrupted, and finite-shot datasets;
- Ridge and MLP baselines plus a loadable, backend-executed hybrid QNN;
- independent node/edge readout calibration and a configurable, N-qubit
  device error atlas (`--qubits`, default 4);
- held-out, multi-seed Grover validation with confidence intervals;
- a second, more realistic worked example: a QROM "phonebook" lookup on a
  5-qubit subgraph, correcting a full 8-entry superposition instead of a
  single Grover target (see [Phonebook QROM demo](#phonebook-qrom-demo));
- an end-to-end Haiqu/local demonstration.

v0.61 generalizes v0.6's hardcoded 4-qubit demo subgraph to an arbitrary
`--qubits N` connected subgraph (`ExecutionConfig.demo_qubit_count`); the
underlying per-qubit/per-edge reconstruction models are unaffected since they
operate on fixed-size diagnostic feature vectors regardless of total device
size.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test]"
```

For real Haiqu execution, install the optional SDK and export the key:

```bash
python -m pip install -e ".[haiqu,test]"
export HAIQU_API_KEY="..."
```

Do not store keys in the repository. Optional IBM credentials use
`IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_INSTANCE`.

## Reproduce

Run all tests:

```bash
pytest -q
```

Generate a quick dataset:

```bash
python scripts/generate_datasets.py --profile smoke
```

Train quick model artifacts:

```bash
python scripts/train_models.py --profile smoke --epochs 3 --patience 2
```

Run the complete local validation without a Haiqu key:

```bash
python scripts/run_end_to_end_demo.py \
  --device aer_simulator \
  --shots 256 \
  --model compare \
  --depths 2 4 8 16
```

Run the dedicated algorithm benchmark with Ridge as the primary reconstruction
and all seven trained QNN inference circuits enabled for comparison:

```bash
python scripts/run_algorithm_benchmark.py \
  --device fake_fez \
  --shots 4096 \
  --model compare \
  --validation-repeats 3 \
  --evaluation-repeats 5 \
  --depths 2 4 8 16
```

Add `--require-cloud` to require an authenticated Haiqu job. Without a key, the
same command is an explicitly labelled local validation and does not claim that
Haiqu mitigation was executed.

Add `--qubits N` to either script to select a larger (or smaller) connected
demo subgraph than the default of 4. This only changes which physical qubits
and edges are characterized and benchmarked; the Ridge/QNN reconstruction
models themselves are unaffected, since they are trained on fixed-size
per-qubit and per-edge feature vectors, not on the total device qubit count.

Use `--device fake_fez` for a noisy local fake backend. The local run is
explicitly labelled `local_fallback`; it does not fabricate Haiqu experiment
IDs or substitute local post-processing for cloud mitigation.

The model choices are:

- `--model ridge`: run only the accurate classical baseline;
- `--model qnn`: use measured trained-QNN predictions for the atlas and
  coherent correction;
- `--model compare`: execute both branches and keep Ridge as the primary
  correction model. This is the default and the recommended pitch mode.

The QNN branch is part of the real end-to-end path. For every reconstructed
channel it performs:

```text
diagnostic counts
-> independently calibrated readout correction
-> physical feature vector
-> saved standardization and learned projection
-> trained, feature-bound QNN circuit on the selected backend
-> measured Z_i and Z_i Z_(i+1) observables
-> saved classical Ridge readout
-> physical alpha, gamma, and kappa parameters
```

The primary Ridge branch uses separate
`ridge_*_readout_corrected.npz` artifacts. They are trained on finite-shot
features after removing the known synthetic assignment-channel bias, rather
than applying a clean calibration to a model that was trained on raw readout
noise. Rebuild these small artifacts with:

```bash
python scripts/train_readout_corrected_ridge.py
```

`results/haiqu/reconstructed_channels.csv` records `model_kind`,
`inference_backend`, and `is_primary`. `results/haiqu/model_deployment.csv`
contains the seven QNN circuits that were actually transpiled and executed.
The generic untrained angle-encoding circuit is no longer used as a substitute
for QNN inference.

For authenticated execution:

```bash
python scripts/run_end_to_end_demo.py \
  --device fake_fez \
  --shots 4096 \
  --model compare \
  --require-cloud
```

Generated tables and figures are written under `results/`; portable datasets,
models, circuit templates, and Haiqu manifests are written under `artifacts/`.

## Algorithm-level correction benchmark

The final benchmark is an exhaustive two-qubit Grover search:

- all four marked states `00`, `01`, `10`, and `11` are tested;
- the suite is repeated on every edge of the selected four-qubit spanning tree;
- each base circuit is folded to requested 2Q depths `2, 4, 8, 16` by
  appending barrier-protected `CX-CX` identities;
- three edges and four depths therefore produce 48 fixed algorithm instances;
- each ideal circuit has one deterministic correct answer.

Before reconstructing the generator, v0.5 executes 20 raw calibration circuits:
two basis states on each of four nodes and four basis states on each of three
edges. The measured assignment matrices are inverted with SVD regularization.
Raw counts remain in the output tables; only a separate calibrated feature view
is supplied to the physical-generator model.

The presentation table and main figure contain four unambiguous modes:

1. raw device execution;
2. independently calibrated readout-only mitigation;
3. validation-gated learned correction;
4. learned correction plus Haiqu advanced mitigation.

The old generator-only probability inverse is excluded from the main figure.
It is retained only in `rejected_generator_ablation.csv`, together with
quasiprobability negativity and simplex-projection diagnostics.

The learned correction propagates the complete reconstructed channel after
every modeled logical 1Q or CX gate. Coherent `alpha`, stochastic `gamma`, and
amplitude-damping `kappa_down` terms therefore all enter an atlas-predicted
Grover response matrix. Its regularized inverse is applied to the measured
distribution and projected back to the probability simplex. No additional
physical correction gates are added to the primary benchmark, avoiding the
extra two-qubit noise that made the earlier end-of-circuit GHZ correction
worse.

`build_grover_search_circuit()` still supports explicit exact coherent inverse
unitaries for controlled studies. Tests verify that those operations are
inserted next to every modeled error location, not collected at the end, but
this higher-gate-count variant is not presented as the recommended hardware
path.

The response inverse is built before any Grover result is seen; Grover counts
are never used to fit its matrices. Independent validation repeats enable or
reject the incremental generator inverse separately for each physical edge and
requested depth. A failed edge-depth pair retains the calibrated readout-only
result.
Final metrics are then computed on different held-out seeds. A conservative
Tikhonov value of `0.03` is fixed before execution to avoid near-pseudoinverse
overcorrection; it is exposed as `--response-regularization`. Each edge gate
requires a positive paired improvement over readout-only and rejects aggressive
inversions when mean quasiprobability negativity exceeds `0.05` or mean simplex
correction exceeds `0.10`.

Means and 95% Student-t intervals are computed across independent held-out
repeat means. Paired TVD improvements and paired success gains compare each
scenario with raw on the same repeat. With only one evaluation repeat the CI
fields are empty, error bars are omitted, and the figure explicitly says that
confidence intervals are unavailable.
Condition numbers, inverse L1 overheads, projection audits, per-edge matrices,
validation rows, and every held-out instance are saved in:

```text
artifacts/haiqu/algorithm_response_models.json
artifacts/haiqu/readout_calibration.json
artifacts/haiqu/execution_audit.json
results/haiqu/algorithm_benchmark_details.csv
results/haiqu/benchmark_seed_summary.csv
results/haiqu/correction_validation.csv
results/haiqu/depth_sweep_benchmark.csv
results/haiqu/depth_sweep_benchmark.png
results/haiqu/presentation_benchmark.csv
results/haiqu/rejected_generator_ablation.csv
results/haiqu/readout_calibration.csv
results/haiqu/final_benchmark.csv
results/haiqu/final_benchmark.png
```

`execution_audit.json` states directly whether QNN inference ran, how many QNN
circuits executed, whether confidence intervals are available, which depth
suite was requested, and whether the learned generator contributed beyond a
readout-only fallback.

This is algorithm-level error cancellation/error mitigation. It is not
fault-tolerant quantum error correction: no syndrome qubits are measured, and
the inverse of irreversible noise is non-CPTP and can amplify shot noise.

## Phonebook QROM demo

`scripts/demo_phonebook_correction.py` is a second, independent worked example
of the same algorithm-level correction methodology, built on a more realistic
circuit shape than Grover: a QROM "phonebook" lookup, following the pattern
from the UCU QML2026 Day 1 data-loading demo.

- 3 index qubits + 2 data qubits on a 5-qubit linear-chain subgraph of the
  selected device;
- `--mode superposition` (default) encodes the full 8-entry phonebook as
  `(1/√8) Σ_i |i⟩|data_i⟩`; `--mode lookup --index <bits>` encodes a single
  deterministic entry instead;
- the same forward-simulated response-matrix machinery as the Grover
  benchmark builds one 4×4 response matrix per index (never using measured
  benchmark counts to fit it);
- a per-index **validation gate**, mirroring the Grover benchmark's per-edge
  gate: the learned correction is enabled for a given index only if held-out
  validation repeats show positive mean improvement, bounded negativity, and
  bounded simplex-projection overhead; otherwise that index silently falls
  back to readout-only correction;
- five comparison modes: raw, readout-only, learned, Haiqu-mitigation-only,
  and learned-plus-Haiqu (the last two require `--require-cloud`).

Run a local smoke test (no Haiqu key required):

```bash
python scripts/demo_phonebook_correction.py \
  --shots 512 \
  --validation-repeats 1 \
  --evaluation-repeats 1
```

Run the full validation-gated demo on Haiqu Cloud:

```bash
python scripts/demo_phonebook_correction.py \
  --shots 4096 \
  --require-cloud \
  --validation-repeats 2 \
  --evaluation-repeats 2
```

`run_phonebook_cloud_demo.ps1` wraps both steps for PowerShell, prompting for
`HAIQU_API_KEY` as a hidden secure-string input (never written to disk).

Outputs land under `results/haiqu/`:

```text
results/haiqu/phonebook_correction_validation.csv   # per-index validation audit
results/haiqu/phonebook_demo_superposition.csv      # final distributions, all modes
results/haiqu/phonebook_demo_superposition.png      # per-mode bar charts, with
                                                      # P(correct) / P(incorrect) sums
```

## Physical conventions

- Liouville matrices use column-major vectorization.
- PTMs use lexicographic unnormalized Pauli words.
- Dataset Choi matrices are normalized to trace one.
- The two-qubit composition convention is `local_before_gate_error`.
- The 2Q regressor predicts the eight gate-specific coefficients; two local
  damping rates are supplied by prior 1Q calibration.

No quantum-advantage claim is made. The QNN is a small deployment demonstrator,
and the classical Ridge model is expected to be the strongest baseline on the
current near-identity synthetic dataset.

The portable `.pt` checkpoints are NumPy NPZ payloads, not pickles.
`HybridQNN.load()` and `PhysicalBaseline.load()` restore them directly.
Exported QASM circuits include measurements and use the same trained angles as
the runtime QNN circuit builder.
