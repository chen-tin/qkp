# Results and metric contracts

Published targets are immutable references, not measured results. A run stores logs, source/protocol/target hashes, interpreter/platform and package versions under its paper's `runs/` directory. Status is `completed` only for the executed command; it does not mean the paper was reproduced.

`infrastructure_smoke` means a bounded software check. `author_data_reanalysis` means recomputing metrics from released author observations without regenerating them. `candidate_reproduction` means a new run that still requires a protocol audit. Only the latter two may be compared by the scorer; neither automatically becomes a verified reproduction.

## Raw-count analysis

Every folder has `analyze.py --input counts.json --output evidence.json`. The analyzer supports binomial error rates, a fitted parity-decay rate from at least three durations, and relative reductions between two independently fitted parity curves. NumPy/SciPy are needed only for fits (`analysis-requirements.txt`). It never divides a shot-error rate by the number of rounds.

For a Bernoulli target, the input is:

```json
{
  "classification": "candidate_reproduction",
  "provenance": {"seed": 0, "checkpoint_sha256": "actual checkpoint hash", "protocol_sha256": "actual protocol hash"},
  "measurements": [{
    "target_id": "copy the registered target id",
    "conditions": {"copy": "the complete target conditions exactly"},
    "method": "binomial",
    "errors": 30,
    "shots": 100000
  }]
}
```

Those example counts are illustrative, not results. For `parity_decay`, replace `errors`/`shots` with `rows`, a list of `{rounds, errors, shots}`. For `relative_parity_decay`, provide `baseline_rows` and `model_rows`. The fitted model is `P(failure | r) = (1 - A*(1-2*epsilon)^r)/2`, with fitted boundary contrast A. This must match the publication's fitting convention before comparing the result. Fit confidence intervals are not automatically supplied: use a trajectory/seed bootstrap and report its design. A point estimate without uncertainty is insufficient for a completed reproduction.

## Scoring externally computed metrics

Threshold, lifetime, and encoding objectives require the paper-specific evaluator. Their output follows this contract:

```json
{
  "paper_id": "the paper ID",
  "classification": "candidate_reproduction",
  "provenance": {"source_commit": "actual revision", "seed": 0},
  "artifacts": [{"path": "raw-results.json", "sha256": "actual SHA-256 of that file"}],
  "observations": [{
    "target_id": "registered target ID",
    "kind": "copy from targets.json",
    "unit": "copy from targets.json",
    "conditions": {"copy": "all registered conditions exactly"},
    "value": 0.0,
    "ci95": [0.0, 0.1]
  }]
}
```

All numbers above are format examples. Artifact paths are resolved relative to the evidence file (absolute paths also work). Include the raw data, configuration, checkpoint, model/source version, split IDs, RNG seeds and analysis code. Hash verification detects changed evidence; it does not validate the implementation that created it.

For threshold targets, retain counts at multiple code sizes and noise values, the finite-size scaling form, fit range and bootstrap draws. For lifetime ratios, retain all logical-state trajectories, physical time units, asymptote convention and both control/intervention fits. For learned encodings, retain coefficients, normalization, overlap, energy, cutoff convergence and both KL residual grids; fewer coherent components alone do not establish better error correction.

For Bernoulli counts, report errors and trials with Wilson intervals (including nonzero upper limits for zero observed errors). For stopping after a target error count, standard fixed-n Wilson intervals are descriptive: use an appropriate sequential procedure for formal coverage. Report shot failure, per-round error, gate failure, frame failure, threshold and squeezing as distinct quantities.
