# Open-system QuTiP experiment: decisions before execution

Status: proposed, not launched, 20 November 2026. These are recommendations for
discussion, not a frozen protocol or a completed reproduction. No new compute
allocation, model change, or experiment is implied by this planning document.

## Scope of the supplied notebook

The supplied [PIQS overview](https://github.com/qutip/qutip-notebooks/blob/master/examples/piqs-overview.ipynb)
studies driven ensembles of identical two-level systems with homogeneous local
and collective emission, dephasing and pumping. Its examples include an
ensemble coupled to a bosonic cavity, transient/steady populations, Wigner
functions and Dicke-basis states. Permutational symmetry compresses the spin
part of the model; it does not automatically compress a GKP oscillator.

Method: N. Shammah, S. Ahmed, N. Lambert, S. De Liberato and F. Nori,
“Open quantum systems with local and collective incoherent processes:
Efficient numerical simulations using permutational invariance,” Physical
Review A98,063815 (2018), [primary article](https://doi.org/10.1103/PhysRevA.98.063815).
The notebook explicitly limits inference about the full Liouvillian spectrum
from its symmetry-restricted construction.

The nbviewer render was inaccessible through the browser fetcher. Its official
raw GitHub notebook was successfully retrieved and only code/markdown cells
were read. Retrieval hash/metadata and stripped source are saved under
results/open_system_planning/piqs_20261120/. No notebook code was executed.

## Decisions and suggested defaults

| Decision | Alternatives | Suggested GKP starting point |
| --- | --- | --- |
| Scientific system and question | Reproduce PIQS spin/cavity examples; validate GKP dynamics; pursue both as separate studies | Ask whether resolving dissipation during syndrome-extraction gates changes logical fidelity/energy versus the current approximation. Treat PIQS reproduction as a separate target. |
| Environment and parameters | Markovian loss/dephasing; thermal excitation; classical correlated displacements; genuinely non-Markovian quantum bath | Start with photon loss and number dephasing separately, then together. Specify collapse operators, rates and clock units explicitly. Choose hardware-derived rates versus controlled stress values; do not label project choices as calibrated. |
| When noise acts | Idle intervals; gates; feedback delay; full cycle | Validate idle channels as controls, then compare simultaneous gate/noise evolution with the current lumped model at matched integrated rates. |
| Measurement and controller access | Unconditional evolution; discrete syndrome extraction and feedback; continuous homodyne monitoring | Retain current discrete syndromes and nearest-lattice feedback first. Keep environmental jumps and true hidden states inaccessible to the decoder. Add frozen Bayes/LSTM only after physical/numerical validation. |
| State and horizon | Oscillator coherent/Fock states; six finite-energy logical cardinal states; one extraction versus repeated memory cycles | Six logical cardinal inputs at existing Delta=.30, one and two cycles initially. Advance toward8cycles only with accepted cutoff, scoring and integration checks. Use simpler analytic states for channel unit controls. |
| Outcome and decision rule | Physical-state agreement; logical fidelity; energy; purity; syndrome distributions | Primary: six-cardinal mean logical fidelity under the existing convention. Secondary: state trace distance, photon number, purity and boundary populations. Proposed practical difference:1percentage point; distinguish equivalent, material and inconclusive confidence intervals. This is a project proposal, not a literature threshold. |
| Runtime and resources | Small bounded CPU pilot; larger trajectory ensemble; EC2 sweep | Agree a pilot time/RAM cap, then select converged cutoffs and fixed trajectory counts from a runtime/convergence pilot. Preserve the existing benchmark and its raw evidence. No new instance or extended runtime is approved by this document. |

## Existing infrastructure and the actual new comparison

Current [channels.py](../quantum/channels.py) already implements open-system
pure-loss and number-dephasing unravelings. [finite_gate.py](../quantum/finite_gate.py)
already calls QuTiP mcsolve for explicit dissipative data/ancilla gates. The
scalable runner uses the separately labeled approximation of applying noise
around an instantaneous gate. Therefore this is not a first use of open-system
physics, and changing the solver's name alone is not an accuracy improvement.

For the current convention, collapse operators are sqrt(kappa)*a and
sqrt(gamma_phi)*n, n=a†a. Loss transmissivity is exp(-kappa*t); number coherence
rho_nm decays as exp[-gamma_phi*t*(n-m)^2/2]. Fix these conventions when mapping
T1/Tphi values or comparing papers. Thermal excitation, additional controls,
Kerr terms, feedback changes and stabilization are separate interventions.
The Lindblad approximation and an externally imposed AR(1) displacement
process are distinct assumptions: Markovian collapse operators do not supply
our original temporal displacement correlation automatically.

Useful controls are zero dissipation, analytically solvable idle loss, and
idle dephasing, followed by loss during the fixed SUM Hamiltonian, dephasing
during it, then both. Match Hamiltonian, input state, integrated rates,
measurement and readout conventions between arms. A difference between two
different physical models is not automatically a software fault. Idle-channel
agreement alone does not validate a noisy driven gate.

## Solver and metric choices the implementation can handle

[mesolve](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-master.html)
evolves the density matrix under a specified Lindblad model and is useful for
bounded deterministic references.
[mcsolve](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-monte.html)
uses stochastic state-vector trajectories, which reduces state-storage cost
for large joint Hilbert spaces but requires ensemble uncertainty estimates.
These target the same Lindblad model when their inputs match; individual
unraveled paths need not coincide.
[smesolve](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-stochastic.html)
is appropriate if continuous measurement-conditioned dynamics becomes the
scientific question. Adopting it would change the current measurement protocol.

A joint density matrix has (N_data*N_ancilla)^2 complex entries. Do not copy the
current4,096/5,120-level production cutoffs into a joint density-matrix run
without a memory/runtime estimate. The existing32/48-level finite-gate controls
failed population gates; they are diagnostics rather than accepted GKP
performance references. All proposed cutoffs remain subject to convergence.

Keep numerical integration error, Fock truncation, Monte Carlo uncertainty and
physical-model mismatch separate. Our pure-target logical fidelity is the
overlap expectation; QuTiP's [fidelity convention](https://qutip.readthedocs.io/en/stable/apidoc/quantumobject.html#qutip.core.metrics.fidelity)
uses the root/Uhlmann convention, so square it where necessary to compare the
same quantity. A decreasing photon number is not sufficient evidence that
logical quantum information is preserved.

## Proposed sequence, pending the decisions above

1. Fix whether the target is the supplied PIQS example or GKP validation, and
   record versioned source, rates, units, measurements and acceptance rules.
2. Run bounded deterministic controls, with an independently reproducible
   control/intervention check before scaling.
3. Compare the same open-system model across implementations, then isolate
   simultaneous versus lumped gate dissipation as the physical approximation.
4. Extend validated settings to decoder comparisons and longer horizons using
   fixed test budgets, convergence gates and properly scoped intervals.

No target curve, GKP performance value or new success result is claimed here.
