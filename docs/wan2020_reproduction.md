# Wan 2020 reproduction: numerical foundations

Registered 18 November 2026. First checkpoint: journal Appendix D tracking
probability and the multiround posterior variance underlying the decoder.
The full finite-energy Figure 3 comparison remains a separate, incomplete
stage until its channel, qubit extraction and worst-case fidelity are matched.

Primary target is [the published PRR article](https://doi.org/10.1103/PhysRevResearch.2.043280),
also [available from Imperial](https://spiral.imperial.ac.uk/bitstreams/848a611d-955e-4b4f-89ae-1a7539925da2/download).
Do not use the arXiv v2 numerical error bounds as journal results: the inspected
preprint says approximately 1e-11 for single-round tracking error, while the
journal's Appendix D says approximately 1e-5. Its multiround/truncation numbers
and Figure 3 were also revised. Local reference copies retain both versions.
Target journal parameters are Delta=.2182 and sigma0²=.0005 in Appendix D;
Figure 3 rounds Delta to approximately .22. No original simulation source or
raw Figure 3 data was found in the paper links and targeted author/code search.
Prior local GKP results use a different displacement/parity model and cannot
substitute for this finite-energy calculation.

## First checkpoint

Question: do independent quadrature and Monte Carlo recover the published
single-round tracking-error scale, and does the Bayesian posterior variance
agree with independent matrix algebra and sampling?

From journal D1, write sqrt(2)xm=n sqrt(pi)+u+e, where u~N(0,sigma0²) and
e~N(0,Delta²) independently. The envelope supplies normalized weights for n;
rounding-boundary crossing modulo four is independent of integer n. Integrate
the boundary-crossing event over u and e using stable Gaussian tail CDFs.
Sum boundaries from -3.5 sqrt(pi) to 3.5 sqrt(pi), integrate |u|<=12 sigma0.
Omitted mass is negligible at the declared parameters. This is a deterministic
reduction of the paper's integral, not an alternative noise model.

Exploratory quadrature before registration gave 1.8026651e-5. The independent
confirmation uses 20,000,000 direct samples, seed191200829, retaining chunk
counts and every failure pair. Report Wilson95 and whether it contains the
quadrature prediction. Agreement with the paper's roughly specified order
means within a factor of three of 1e-5; this is not agreement to a stated exact
number. A discrepancy beyond the Monte Carlo interval triggers an implementation
audit. Smoke uses100,000 and is not evidence at this rare-event scale.

For the Gaussian posterior component, draw200,000 independent trajectories,
200rounds, seed191200830. Control: unknown displacement theta_h without an
estimator. Intervention: conditional Gaussian mean using all measurements.
The recurrence is theta_h=(theta_(h-1)+u_h)/2 and F_h=2theta_h+e_h, after
removing known measurement-induced steps and choosing the local peak. This is
the paper's **local Gaussian approximation**, not an exact oscillator/channel
simulation. It deliberately isolates Eqs13/15 from tracking/truncation errors.

Compare three ways: Kalman recursion, direct inverse precision matrix of
EqC10 (arXiv Eq43), and Monte Carlo residual mean square error. Compare Eq15
as the published truncated expansion, not as an exact formula. Success requires
matrix/recursion absolute difference<1e-12 and final simulated MSE within its
95% chi-square interval around exact posterior variance. Failures of these
checks contradict implementation consistency; nonfinite runs are infrastructure
failures. Smoke uses1,000 trajectories and10rounds.

This stage does not establish a logical fidelity, GKP lifetime, memoryless
decoder performance, or reproduction of Figure3. Those require the actual
finite-energy channel and the paper's modular-subsystem/worst-case scoring.
Do not replace these with binary logical-error counts from our current site.

## Execute

An independent validator runs the documented no-estimator control and Bayesian
intervention smoke before production Monte Carlo:

```sh
.venv/bin/python -m pytest tests/test_wan2020_reproduction.py -q
.venv/bin/python -m experiments.reproduce_wan2020 --smoke
.venv/bin/python -m experiments.reproduce_wan2020
```

Results, raw terminal samples, hashes, plan and diagnostic curves are isolated
under `results/reproduce_wan2020/`. Test seeds, model equations and thresholds
are fixed before confirmation. Production results remain distinct from smoke.
