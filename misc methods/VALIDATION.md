# Infrastructure validation — 18 November 2026

These checks validate infrastructure only. No full paper reproduction was completed in this task.

| Check | Result |
|---|---|
| Registry, source/target contracts, units/conditions, evidence hashes, smoke rejection, Wilson bounds, parity-decay fit | 8 tests passed in the isolated Python 3.11 environment |
| Independent documented check | Default Python: 6 pass + 1 optional SciPy skip; isolated environment: all 7 pass |
| nbp4 smoke | Passed; `infrastructure_smoke` |
| resgru smoke | Passed; `infrastructure_smoke` |
| tcn smoke | Passed; `infrastructure_smoke` |
| mamba replay | Passed; `author_data_reanalysis` |

The independent validator also built and ran the documented NBP4 smoke: 128 sampled frames. Those smoke counts are not paper-performance evidence.

ResGRU and TCN each use the author circuit, a seeded 64-shot input, one 32-shot gradient step and 32 held-out shots. The naive constant-zero prediction is retained as a control. This tests data layout and differentiability, not decoder accuracy.

Mamba replay recomputed four released d=3, r=28, p=.002 simulated outcome sets (500,000 shots each). With injected decoder noise: Mamba 29,761 errors, Transformer 31,568. Without injected noise: Mamba 29,529, Transformer 31,361. These reproduce arithmetic on author data, not training or hardware per-round LER.

## Resolved setup defects

- The existing qkp environment lacks Stim/Brevitas; dependencies were installed in `misc methods/.smoke-venv` instead.
- Upstream TCN NumPy 2.2.6 conflicts with Brevitas 0.12.1. Smoke uses NumPy 1.26.4.
- SciPy 1.15.3 failed to load on this macOS host. Smoke uses SciPy 1.16.3.
- The NBP4 bridge creates a serial C++17 copy, changes code parameters to the registered toric anchor, and saves the complete patch. It does not modify upstream files.

## Remaining scope

Six folders currently provide source retrieval, protocols and scoring without a model runner. Missing code, coefficients, dependency integration or paper-era configuration reconciliation is listed in each folder. Feedback-GRAPE has an author-example launcher but was not launched or validated here. The full NBP4 anchor is runnable but was not launched.

Existing long-running benchmark sources and environments were not changed. Exact successful local run paths and measured smoke/replay observations are retained in [validation-summary.json](validation-summary.json). The resolved smoke package versions are in [smoke-environment.lock.txt](smoke-environment.lock.txt).

Independent review caught the generic C++ ep0=0.1 initialization. The toric training script sets ep0=0.37 at m=3n; the adapter and target conditions now use 0.37. Earlier smoke runs used 0.1 and are not the final configured anchor.

The independent scorer audit also found that out-of-domain probabilities were accepted. Metric-specific value/CI bounds now reject them, with a regression test; negative relative improvements remain valid. The author <= stopping condition is accounted for by passing limits one below the intended error/word caps. The corrected NBP4 smoke was rerun successfully.
