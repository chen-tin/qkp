# Next quantum-reference stage

Registered before the larger-sample eight-cycle continuation, 19 November 2026.

The immediate question is whether the original eight-cycle nearest-lattice
reference is stable between3072 and4096 Fock levels after increasing the
sampling budget from64 to512 trajectories per cardinal input state. Physical
operators, data/ancilla states, noise, decoder policy, seed and scoring remain
the protocol in `docs/qutip_reference.md`. This changes sampling precision
only; it is not a new physical model or decoder training run.

The independent control/intervention check and explicit two-mode QuTiP tests
already cover this evolution. Boundary-population, norm and extraction gates
remain unchanged. Compare paired fidelity differences at every cycle, using
the registered +/- .01 numerical-equivalence band. If an interval fits wholly
inside the band and all population gates pass, that cycle is provisionally
stable; if it excludes the band, investigate numerical bias; otherwise it is
inconclusive. Do not infer failure of the decoder from a numerical failure.
This fixed continuation adds one budget of512 trajectories per input state,
with no automatic unbounded growth. Subsequent stages remain Planned.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.qutip_reference \
  --convergence --cutoffs 3072 4096 --rounds 8 --per-state 512 \
  --points nominal --decoders nearest
```

The follow-up physics stage will introduce photon loss and oscillator
dephasing separately, then together, under the same measurement protocol.
`quantum/channels.py` implements exact pure-loss Kraus trajectories and
Lindblad number-dephasing random rotations. Their independent tests compare
with QuTiP `mesolve`. These channels are not yet used in the reference
benchmark, so no loss/dephasing performance result is claimed.

For loss, eta=exp(-cycle_time/T1); the environment's loss count is hidden from
the decoder. For dephasing, strength=gamma_phi*cycle_time with collapse
operator sqrt(gamma_phi)*n. The convention is explicit because reported Tphi
definitions can differ by factors of two. Validate identity/noise-free limits
and Fock convergence before production runs. Avoid adding physical loss on
top of a displacement variance already derived from that same loss mechanism.

Hardware comparison will use primary-source public data where available,
document protocol and parameter mappings, fit on a calibration subset and
evaluate separate observables or held-out records. The existing Sivak
parameter mapping is a rough anchor, not validated hardware agreement. New
laboratory measurement collection requires laboratory access. No external
messages, paid compute or Solar writes are part of this local continuation.
