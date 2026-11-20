# Published decoder reproduction runs

Three separate paper targets were requested on18November2026. They do not
validate the completed GKP noise-grid benchmark by themselves. The current
interactive report is writeup/artifact/gkp-walkthrough/reproductions.html,
linked from the main walkthrough. It updates from saved artifacts while
`python -m writeup.artifact.build_reproduction_report --watch` runs.

| Paper | First target | State at launch of the live report |
| --- | --- | --- |
| Convy2022 | Figure2 SchemeA, gamma=.04/us; Bayes and2×32LSTM | Bayes83.98% vs published88.68%, discrepancy unresolved;3diagnostic LSTM seeds running |
| Baireuther2018 | Figure3 Surface-17 Pauli circuit,2branches each2×64LSTM | Source-equivalence checks passed;4million training histories generated;3seeds running |
| Wan2020 | Published AppendixD tracking and Bayesian variance; Figure3 scoring audit | Component checks pass; full Figure3 not reproduced |

## Current production artifacts

- Convy diagnostic continuation: `results/reproduce_convy2022/anchor/bfcba380fc8e8c55/`.
  Original Bayesian control: `results/reproduce_convy2022/anchor/a842b65b9b51b2eb/`.
  Log: `results/reproduce_convy2022/production.log`.
- Baireuther: `results/reproduce_baireuther2018/figure3/aaacc7ab3e3a280d/`.
  Log: `results/reproduce_baireuther2018/production.log`.
- Wan components: `results/reproduce_wan2020/components/91b306a688bce429/`.
  Tracking356/20000000=1.78e-5, Wilson95[1.60445e-5,1.97475e-5].
  Independent quadrature1.80266511765e-5. Posterior-variance confirmation
  uses200000trajectories over200rounds; matrix/recursion agree within5.5e-20.
- Wan fidelity audit: `results/reproduce_wan2020/fidelity_audit/ea2e740f9366feba/`.
  Initial worst fidelity.98164836; at200rounds no-QEC plus-state.91163403,
  all-pure-state minimum.90193657. All41plus-state points agree with the
  digitized journal curve within.0003. This leaves an unresolved scoring
  convention because the published plot is labeled F_worst.
- Live report log: `results/reproduction_report.log`.

## Resume if interrupted

Check for existing processes before launching another copy. Do not write to
an active run from a duplicate trainer. Each trainer resumes its latest
completed epoch. The source, data, and protocol hashes are part of run identity;
do not edit frozen source/protocol files while a production run is active.

```sh
ps -axo pid,etime,command | rg 'experiments.reproduce_|build_reproduction_report'
.venv/bin/python -u -m experiments.reproduce_convy2022 --device cpu
.venv/bin/python -u -m experiments.reproduce_baireuther2018 --device mps
.venv/bin/python -u -m writeup.artifact.build_reproduction_report --watch
```

Run the trainers in separate terminals if resuming both. The report updater
reads files only; it does not train or select models. It refreshes every30s
while a trainer is active, marks an interrupted process honestly, and refreshes
the walkthrough ZIP once both trainer processes have exited.

Convy stops after its registered validation patience or100epochs and evaluates
each seed separately. Its comparison remains inconclusive while the Bayesian
control mismatch is unresolved. Baireuther has no production epoch cap; it
stops only after100epochs without validation improvement per seed, selects the
seed by validation and evaluates the300-cycle test once. This can take hours.
Do not mistake a validation curve or a saved best checkpoint for a finished
reproduction. The final report includes test comparisons automatically as
those output files appear.

## Scope remaining

Full Convy agreement needs resolution of the Bayesian-control discrepancy and
unavailable original neural/evaluation notebook. The three prescribed
training runs still provide an internal Bayes-vs-LSTM diagnostic without
changing parameters to match the paper.

Wan's full finite-energy memoryless and memory-assisted Figure3 channels and
failure corrections are not implemented. Its Gaussian posterior checks do
not establish a logical lifetime. Its published journal differs materially
from arXivv2; journal sources and the explicit scoring-convention audit must
remain visible alongside all comparisons.

The other figures, noise sweeps and hardware models in these papers are
outside these first targets. Primary source copies, pinned author commits,
independent validation records and individual registered protocols are in
`reproductions/` and `docs/*reproduction*.md`.
