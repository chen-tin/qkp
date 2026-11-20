# Research recordkeeping

This repository supports a future research paper. Maintain [RESEARCH_NOTES.md](RESEARCH_NOTES.md) as the canonical, paper-oriented notebook for work performed here. Read its current status, corrections, and open questions before research, simulation, decoder, benchmark, or analysis work. Supporting protocols and detailed findings remain in docs/ and the immutable run directories.

## Required updates

- Log every substantive work item: implementation, literature investigation, experiment plan, launch, interruption, completed run, analysis, bug, negative result, validation, and decision. Consolidate routine actions into one entry per coherent work item; avoid a transcript of tool calls.
- Update the notebook before handing work back to the user. For long-running work, record the launch and exact run identity while it is active, then append the outcome when available. Never mark a launch, smoke check, checkpoint, or partial grid as a completed benchmark.
- Use dated entries with: question/purpose; status; method and citation; configuration and data provenance; metric and decision rule; evidence paths; result and uncertainty; limitations; decision and next step. Use the template in RESEARCH_NOTES.md.
- Always record actual final noise parameters and units, distribution and temporal structure, rounds, trial counts, independent training seeds, split identities, checkpoint/source/config hashes or the manifest containing them, and environment/device when relevant. Link existing manifests rather than copying large arrays or every hash into prose.
- Define the metric, its direction, denominator, horizon, aggregation and uncertainty. Keep shot failure, cumulative logical error, parity-decay error per round, logical-state fidelity, conditional-state agreement with QuTiP, and hardware accuracy distinct. Deterministic quadrature errors do not acquire Monte Carlo confidence intervals.
- Separate published targets, our measured results, derived formulas, hypotheses, infrastructure smoke checks, author-data reanalysis, and verified reproductions. A metric match alone does not establish physical/protocol equivalence. Report failed controls, incomplete coverage, all registered seeds and numerical-gate failures.
- Attach primary-paper links and equation/figure/table or source-code locators when known. Label unverified references and reconstruction choices. Do not invent bibliographic details or numerical results; verify uncertain citations before using them as support.
- Preserve the original user notes in docs/user_research_notes_2026-11-19.md. Retain their ideas and distinguish later corrections from their original wording.
- Add dated corrections that identify what they supersede. Update the notebook's current-status table, but do not silently erase earlier conclusions. Do not edit frozen protocols, source snapshots, checkpoints or raw evidence to repair a citation or improve agreement. Correct active documentation/report generators and record the correction separately.
- Link the relevant report, figure, raw result, analysis script and reproduction command. Record which checks actually ran, with their scope; never present an old test count as a fresh test result.
- Keep paper-facing claims within the demonstrated model, noise coverage, horizon and numerical validity. In particular, a classical displacement study does not validate finite-energy hardware, and a high conditional-state fidelity against QuTiP does not establish decoder superiority.

## Record locations

- RESEARCH_NOTES.md: canonical overview, methods, sources, dated work log, decisions and paper gaps.
- docs/findings.md: existing historical findings; retain it and link relevant entries.
- docs/*experiment*.md and reproduction protocols: pre-run methods; preserve frozen copies.
- results/<study>/<profile>/<digest>/: run-specific manifests, raw observations and analyses.
- misc methods/: paper-specific reproduction infrastructure; distinguish readiness from executed reproduction.
- writeup/artifact/: HTML figures/reports and their generators.

This recordkeeping instruction does not authorize new compute-heavy experiments, publication, external messages, or changes beyond the user's task. Use the existing task authorization and continue maintaining the notebook within that scope.
