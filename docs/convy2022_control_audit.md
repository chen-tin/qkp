# Convy control audit and diagnostic training continuation

Recorded 18 November 2026 after the predeclared Bayesian-control evaluation,
before any production LSTM training. Original protocol and code remain frozen.
This separate note resolves the original instruction to investigate a failed
control before proceeding; it does not make that control compatible.

At gamma=.04/us the final exact-state Bayesian accuracy is25194/30000=.8398,
Wilson95 [.83560597,.84390702]. Figure2 SchemeA is approximately.8868
(vector digitization, tolerance.0002). The gap is4.7percentage points and
exceeds the registered compatibility criterion. Neither the LSTM nor the
whole paper may be described as reproduced from this run.

Checks completed:

- An independent validator applied the authors' Gaussian likelihood and
  filtering recurrence to all30000 trajectories. All final predictions match
  our implementation; six intermediate float-rounding differences among18.75M
  predictions do not alter the result.
- Empirical measurement variance6.6483/6.6478 matches1/(4.7*.032). Measured
  single-qubit flip rates.04036/.04004/.04007 match the specified.04/us.
  The test contains3750trajectories per initial state.
- The authors' fitted transition estimator plus first-label initialization
  gives.8398667, so that convention does not resolve the discrepancy.
- The source's SchemeA selects equal resonator variances, no correlations,
  no transient or drift. Its negative signal convention is a global sign
  reversal. The published journal figure and methods agree with the preprint
  at this target, unlike the material version changes in Wan.
- Mean-over-time accuracy.8919406 and an added final majority recovery.8908
  are closer to the plotted point, but neither is the exact-final-state
  score in the accessible authors' tracking helper. We do not substitute
  either metric or tune noise values to force agreement.

Evidence: results/reproduce_convy2022/anchor/a842b65b9b51b2eb/
validation_author_filter.json, validation_data_and_metrics.json and
validation_smoke_audit.json. The original neural Colab remains unavailable
(404 through Drive/public export). Missing original training/evaluation code
prevents resolving the remaining scoring/simulation ambiguity confidently.

Continuation decision: execute all three originally registered LSTM seeds,
with unchanged architecture, data, optimizer, loss, selection and stopping
rules, as a **diagnostic methods reproduction**. This evaluates whether the
network can approach our independently verified Bayesian control; it cannot
establish compatibility with the published experiment while the control
mismatch remains unresolved. Use the already fixed test data without tuning
hyperparameters against it. Report every seed, the original published target,
and the failed control together. Do not select a favorable metric or seed.

For CPU execution the device is recorded as a new run identity. Data streams
and hashes remain the same; immutable arrays may be hard-linked from the
original control run to avoid duplicating them. This is a hardware change,
not a change to the stochastic model or training protocol. The existing
independent control/intervention smoke and architecture checks remain valid.
