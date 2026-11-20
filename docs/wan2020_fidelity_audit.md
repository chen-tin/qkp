# Wan Figure 3: finite-energy scoring audit

Registered 18 November 2026 after an exploratory calculation, before the
41-point comparison. This is a diagnostic extension of the components
protocol, not an amendment to its frozen run. It does not implement the
memoryless or memory-assisted finite-energy syndrome extraction circuit.

Use Wan's published Eq.1 with Delta=kappa=.2182 and the Gaussian displacement
channel Eq.2. Extract a qubit using the position-bin subsystem decomposition
in Pantaleoni, Baragiola and Menicucci, PRL125,040501 (2020), Eqs.1/3:
rho_L[l,l'] = sum_m integral_{-sqrt(pi)/2}^{sqrt(pi)/2}
rho((2m+l)sqrt(pi)+u,(2m+l')sqrt(pi)+u) du.
Primary references are the published Wan PDF and
https://arxiv.org/html/1907.08210v3 .

Integrate the displacement distribution and Gaussian peaks analytically.
Finite sums use peaks s=-12..12 and bins m=-20..20; repeat selected endpoints
with peaks -16..16 and bins -24..24 to check truncation below 1e-10.
The small nonorthogonality of the approximate GKP codewords is removed with
the symmetric inverse square root of their Gram matrix, defining an exact
isometric qubit encoding. Retain that explicit convention rather than silently
assuming orthogonality. It changes these results by much less than figure
digitization precision.

Compare the logical |+> input to the minimum over the entire pure-qubit Bloch
sphere. Optimize from the six Pauli directions, retain minimizers, and check
the resulting channel is trace preserving, Hermiticity preserving and
completely positive. Independently compare the analytic extraction at zero
and nonzero displacement variance with numerical quadrature of the original
wavefunctions. This checks formulas without fitting to Figure3.

Exploratory values before registration: initial minimum fidelity .98164836;
at M=200 and sigma0²=.0005, |+> fidelity .91163403 versus all-state minimum
.90193657. The published no-QEC curve visually ends near .912 despite the
axis label F_worst. Thus agreement of |+> with the plot does not establish
agreement for the worst case. We have not located a stated restriction to real
qubit amplitudes or original code resolving this convention. Treat this as an
unresolved scoring question, not a demonstrated error by the authors.

Extract journal Figure3 blue pixels, calibrating x=0 at87 and x=.1 at998,
y=.88 at857 and y=.96 at284 in the original1015x934 embedded JPEG. Ignore
the inset and text by selecting the cyan color; report a conservative .0003
absolute digitization tolerance. Grid: rounds0,5,...,200. Do not change the
state, width, or channel variance to force the worst-case curve to match.

Execute tests and smoke before the comparison:

```sh
.venv/bin/python -m pytest tests/test_wan_fidelity.py -q
.venv/bin/python -m experiments.wan_fidelity_audit --smoke
.venv/bin/python -m experiments.wan_fidelity_audit
```

The uncorrected state is the control; the Gaussian residual variance from the
Bayesian estimator is a separate diagnostic intervention. Its extracted
fidelity omits tracking/truncation failures and is not the memory-assisted
Figure3 curve. No full reproduction is claimed from this audit.
