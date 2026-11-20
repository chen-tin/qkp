# QuTiP finite-energy reference comparison

Registered 19 November 2026 before the first physics runs. This bounded experiment asks whether decoder performance in the displacement simulator survives finite-energy data and ancillas with explicit conditional syndrome extraction. Researchers transferring a fast decoder to a bosonic circuit can use the result to identify where the approximation needs improvement.

The previous QuTiP checks in `experiments/benchmark_finite_energy.py` cover state preparation, single-round readout and truncation. They do not establish multiround decoder performance. Wan's separate fidelity audit leaves the full finite-energy protocol incomplete. Existing displacement-heatmap results remain a separate baseline.

## Physical model and controls

Conventions: `[q,p]=i`, `L=sqrt(pi)`. The data encode six cardinal logical states in the symmetrically orthonormalized span of `exp(-Delta^2 n)|mu_ideal>`, with `Delta=.30`. QuTiP constructs the oscillator q/p operators; their eigendecompositions apply unitary displacements and conditional Kraus operators efficiently in batches.

For q extraction a fresh finite-energy logical-plus GKP ancilla interacts by the ideal SUM gate `exp(-i q_data p_ancilla)` and is measured by ideal q homodyne. The conditional data operator is `M_y=phi_plus(y-q_data)`. The p extraction is its Fourier conjugate, using a fresh resource with the same measured-quadrature wavefunction. The full outcome y conditions the state; the classical decoder receives only `s=wrap(y,L)`. The ancilla peak probability standard deviation equals sigma_m: its heat-kernel parameter is `t=atanh(2 sigma_m^2)`. This matches local peak widths, not an entire Gaussian measurement channel.

The Mehler kernel evaluates the resource wavefunction, avoiding a second oscillator in every trajectory. Ancilla quadrature sampling uses normalized piecewise-constant amplitude bins of width dx=.003. The sampler/update share that same density. Drawing a data-quadrature eigenvalue is only an auxiliary mixture algorithm to sample y; it never projects the conditional data ket onto that eigenvalue. A direct two-mode QuTiP unitary check tests the reduced Kraus implementation.

Steane extraction with SUM/inverse SUM and GKP resources is described in [Noh, Chamberland and Brandao, PRX Quantum 3, 010315 (2022), Sec. II B](https://doi.org/10.1103/PRXQuantum.3.010315). This is a local finite-energy reference experiment, not a reproduction of their surface-code thresholds. The logical modular-subsystem extraction follows the same convention as `experiments/wan_fidelity_audit.py`.

Three arms share exogenous q/p displacement draws:

1. Classical control: independent AR(1) q and p residuals, Gaussian readout, parity corrections; this is the two-quadrature extension of our fast model. Each marginal must match `Simulator` exactly.
2. Finite-energy replay control: apply the classical controller's displacements to the finite-energy ket, without syndrome-extraction backaction. This isolates encoding/readout and displacement-cutoff effects.
3. Finite-ancilla reference: explicit conditional q measurement, q correction, p measurement, p correction, each cycle. This adds a physical measurement instrument and its backaction. Comparing this arm with replay is a combined finite-ancilla/measurement intervention; it cannot attribute an effect to one microscopic mechanism.

Each correction uses the shortest representative of `s+jL` modulo `2L`. It has the same ideal-code action as the existing simulator; on finite-energy states this choice matters and is fixed before running. The data displacement in each cycle is q followed by p; their Weyl global phase is irrelevant in the untruncated limit. Initial fidelity at cycle zero is reported rather than silently renormalized to one.

## Settings and hypothesis

Noise points `(sigma_q/L,sigma_m/L,rho)` are nominal `(.20,.05,.90)`, white `(.20,.05,0)` and high readout `(.20,.16,.90)`. Both quadratures have the same parameters and independent innovations. Controllers are nearest lattice, Bayes assuming the stated Gaussian AR(1) model (not a quantum oracle), and frozen size-sweep checkpoint `h256_l3_s0.pt`. Both quadratures use separate controller histories. There is no retraining on reference outcomes, and no seed selection. This is an exploratory transfer check of one LSTM seed.

Before observation we expect the finite-ancilla arm may change six-state average logical fidelity by at least one percentage point within eight cycles; the sign and controller ranking are not assumed. No prediction of hardware accuracy is made.

The primary output is the six-state mean logical fidelity after each cycle, computed by tracing the modular gauge subsystem. Record each state's fidelity, mean photon number, Fock-boundary population, all syndrome/correction arrays, noise and hashes. Ideal classical fidelity counts q-parity flips as X errors and p-parity flips as Z errors; Y-state errors are their XOR. Fidelity observations are conditional probabilities, not sampled Bernoulli bits. Report stratified independent-trajectory standard errors and pointwise normal 95% intervals; shared random draws permit paired arm differences. These are exploratory intervals without multiplicity adjustment. Fresh confirmation is required for a claim about model superiority.

## Numerical gates and execution

First run unit checks and an independent documented control/intervention smoke. Smoke uses four trajectories per input state, two cycles and cutoff96 and cannot support physical conclusions.

Convergence runs use 64 trajectories per cardinal state, eight cycles and cutoffs192,256,384. Production uses 256 per state, eight cycles and a cutoff384, provided diagnostics allow it. If necessary use a larger cutoff or restrict the valid horizon, recording all failed numerical gates. A converged result needs: maximum norm error below1e-8, logical-extraction trace error below1e-7, mean population in the top10% of the Fock basis below1e-4 at every cycle, and maximum individual top10% population below.01. In addition compare mean fidelity at the two largest cutoffs: require the paired 95% interval for the difference to lie within +/- .01. Repeat the largest-cutoff diagnostic with dx halved; require its interval to lie within +/- .005. If intervals are too wide, increase convergence samples before asserting equivalence. All changes are recorded in the command/config identity.

Independent review before scaling also requires refinement of logical-extraction quadrature: trace preservation alone does not certify its off-diagonal integrals. The default is192 bins per modular cell; repeat scoring with384 bins and require paired fidelity differences within +/- .001. An initial cutoff768 setup exposed Gaussian-seed underflow in the Hermite recurrence; the recurrence now uses column log scaling. These are numerical setup diagnostics, not physics observations. Larger-cutoff exploration starts at768 and1536 because the first192/256/384 runs failed the population gate by large margins. The valid horizon can be shorter than eight cycles and is reported explicitly.

After these numerical gates, a physical difference is supported when a 95% interval for the reference-minus-classical mean is entirely beyond +/- .01; equivalence within one percentage point requires the whole interval inside that band. Otherwise the comparison is inconclusive. A failed cutoff gate makes that horizon numerically unresolved, regardless of a large apparent difference. Negative and null results are retained.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_qutip_reference.py -q
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.qutip_reference --smoke --points nominal --decoders nearest
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.qutip_reference --convergence --points nominal --decoders nearest
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.qutip_reference
```

Results are isolated in `results/qutip_reference/<profile>/<digest>/`; completed cells can resume within an identical configuration. Full source, checkpoint, interpreter and dependency versions are part of each run identity. CPU/BLAS threads are limited to one so the existing MPS reproduction can continue. Standard `.venv` already contains QuTiP; no environment changes are required.

This initial reference excludes photon loss, dephasing, finite gate duration, ancilla faults and a calibrated hardware measurement chain. Those are later physical interventions. It does not complete a50-round,61-setting quantum benchmark or establish the same noise family as the original simulator. Status before execution: Planned.
